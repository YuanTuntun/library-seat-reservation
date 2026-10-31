#!/usr/bin/env python3
import argparse
import ctypes
import json
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageGrab
from rapidocr_onnxruntime import RapidOCR

LOG = logging.getLogger("seat-reservation")
BASE_DIR = Path(__file__).resolve().parent


class ReservationError(RuntimeError):
    pass


class SeatUnavailable(ReservationError):
    pass


@dataclass(frozen=True)
class WindowState:
    window_id: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class OcrItem:
    text: str
    score: float
    box: tuple

    @property
    def center(self):
        return (
            sum(point[0] for point in self.box) / 4,
            sum(point[1] for point in self.box) / 4,
        )


@dataclass
class Snapshot:
    state: WindowState
    image: Image.Image
    items: list

    def combined_text(self):
        return normalize(" ".join(item.text for item in self.items))


class ClientData(ctypes.Union):
    _fields_ = [("b", ctypes.c_char * 20), ("s", ctypes.c_short * 10), ("l", ctypes.c_long * 5)]


class ClientMessage(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", ClientData),
    ]


class XEvent(ctypes.Union):
    _fields_ = [("client", ClientMessage), ("pad", ctypes.c_long * 24)]


class X11:
    def __init__(self):
        self.display_name = os.environ.get("DISPLAY", ":0")
        self.x11 = ctypes.CDLL("libX11.so.6")
        self.xtst = ctypes.CDLL("libXtst.so.6")
        self._set_signatures()
        self.display = self.x11.XOpenDisplay(self.display_name.encode())
        if not self.display:
            raise ReservationError(f"无法打开显示 {self.display_name}")
        root_info = self._run(["xwininfo", "-root"])
        self.root_id = int(re.search(r"Window id:\s+(0x[0-9a-f]+)", root_info).group(1), 16)

    def _set_signatures(self):
        self.x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x11.XOpenDisplay.restype = ctypes.c_void_p
        self.x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self.x11.XInternAtom.restype = ctypes.c_ulong
        self.x11.XSendEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_long,
            ctypes.POINTER(XEvent),
        ]
        self.x11.XFlush.argtypes = [ctypes.c_void_p]
        self.x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self.xtst.XTestFakeMotionEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        self.xtst.XTestFakeButtonEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_ulong,
        ]

    @staticmethod
    def _run(args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)

    def close(self):
        if self.display:
            self.x11.XCloseDisplay(self.display)
            self.display = None

    def find_seat_window(self):
        process_info = self._run(["ps", "-C", "WeChatAppEx", "-o", "pid=,args="])
        pids = {
            int(line.split(None, 1)[0])
            for line in process_info.splitlines()
            if line.strip() and "--type=" not in line
        }
        if not pids:
            raise ReservationError("未找到微信预约页面进程，请先打开座位管理系统")
        tree = self._run(["xwininfo", "-root", "-tree"])
        pattern = re.compile(r'^\s*(0x[0-9a-f]+)\s+"[^"]*":\s+\([^)]*\)\s+(\d+)x(\d+)', re.M)
        candidates = []
        for match in pattern.finditer(tree):
            window_id = int(match.group(1), 16)
            width, height = int(match.group(2)), int(match.group(3))
            if width < 800 or height < 600:
                continue
            try:
                pid_info = self._run(["xprop", "-id", str(window_id), "_NET_WM_PID"])
            except subprocess.CalledProcessError:
                continue
            pid_match = re.search(r"=\s*(\d+)", pid_info)
            if pid_match and int(pid_match.group(1)) in pids:
                candidates.append((width * height, window_id))
        if not candidates:
            raise ReservationError("未找到微信座位预约窗口，请打开系统后再运行")
        return max(candidates)[1]

    def geometry(self, window_id):
        info = self._run(["xwininfo", "-id", str(window_id)])
        values = []
        for label in ("Absolute upper-left X", "Absolute upper-left Y", "Width", "Height"):
            match = re.search(rf"{label}:\s+(-?\d+)", info)
            if not match:
                raise ReservationError("无法读取预约窗口位置")
            values.append(int(match.group(1)))
        return WindowState(window_id, *values)

    def activate(self, window_id):
        active = self._run(["xprop", "-root", "_NET_ACTIVE_WINDOW"])
        match = re.search(r"window id #\s+(0x[0-9a-f]+)", active)
        if match and int(match.group(1), 16) == window_id:
            return
        event = XEvent()
        event.client.type = 33
        event.client.display = self.display
        event.client.window = window_id
        event.client.message_type = self.x11.XInternAtom(self.display, b"_NET_ACTIVE_WINDOW", 0)
        event.client.format = 32
        event.client.data.l[0] = 2
        self.x11.XSendEvent(self.display, self.root_id, 0, (1 << 19) | (1 << 20), ctypes.byref(event))
        self.x11.XFlush(self.display)
        time.sleep(0.35)

    def capture(self, window_id):
        state = self.geometry(window_id)
        image = ImageGrab.grab(
            bbox=(state.x, state.y, state.x + state.width, state.y + state.height),
            xdisplay=self.display_name,
        ).convert("RGB")
        return state, image

    @staticmethod
    def _ensure_unchanged(expected, current):
        if expected != current:
            raise ReservationError("预约窗口在操作前发生移动或缩放，正在重试")

    def click(self, snapshot, point):
        current = self.geometry(snapshot.state.window_id)
        self._ensure_unchanged(snapshot.state, current)
        x = round(current.x + point[0])
        y = round(current.y + point[1])
        if not (current.x <= x < current.x + current.width and current.y <= y < current.y + current.height):
            raise ReservationError("识别到的点击位置超出预约窗口")
        self.xtst.XTestFakeMotionEvent(self.display, -1, x, y, 0)
        self.xtst.XTestFakeButtonEvent(self.display, 1, 1, 0)
        self.x11.XFlush(self.display)
        time.sleep(0.06)
        self.xtst.XTestFakeButtonEvent(self.display, 1, 0, 0)
        self.x11.XFlush(self.display)


class Ocr:
    def __init__(self):
        self.engine = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=2)

    def read(self, window_id, x11):
        state, image = x11.capture(window_id)
        result, _ = self.engine(np.asarray(image))
        items = [
            OcrItem(str(text), float(score), tuple(tuple(point) for point in box))
            for box, text, score in (result or [])
        ]
        return Snapshot(state, image, items)


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", "", text).replace("：", ":").replace("–", "-").replace("—", "-")


def in_region(item, region):
    if region is None:
        return True
    x, y = item.center
    return region[0] <= x < region[2] and region[1] <= y < region[3]


def find_items(snapshot, text, contains=False, region=None):
    target = normalize(text)
    matches = []
    for item in snapshot.items:
        value = normalize(item.text)
        matched = target in value if contains else target == value
        if matched and in_region(item, region):
            matches.append(item)
    return sorted(matches, key=lambda item: (item.center[1], item.center[0]))


def has_text(snapshot, text, contains=False, region=None):
    return bool(find_items(snapshot, text, contains, region))


class Reservation:
    def __init__(self, config, dry_run=False):
        self.config = config
        self.dry_run = dry_run
        self.tz = ZoneInfo(config["timezone"])
        self.target_date = datetime.now(self.tz).date() + timedelta(days=int(config.get("date_offset", 1)))
        self.state_path = BASE_DIR / "state.json"
        self.log_dir = BASE_DIR / "logs"
        self.x11 = X11()
        self.window_id = self.x11.find_seat_window()
        self.ocr = Ocr()
        self.delay = float(config.get("action_delay_ms", 450)) / 1000

    def close(self):
        self.x11.close()

    def read(self):
        self.x11.activate(self.window_id)
        return self.ocr.read(self.window_id, self.x11)

    def pause(self):
        time.sleep(self.delay)

    def click_item(self, snapshot, item):
        self.x11.click(snapshot, item.center)
        self.pause()

    def click_text(self, snapshot, text, contains=False, region=None, occurrence=0):
        matches = find_items(snapshot, text, contains, region)
        if len(matches) <= occurrence:
            raise ReservationError(f"页面上未找到可点击文字：{text}")
        item = matches[occurrence]
        point = item.center
        value = normalize(item.text)
        target = normalize(text)
        index = value.find(target) if contains else -1
        if index >= 0 and value != target:
            min_x = min(point[0] for point in item.box)
            max_x = max(point[0] for point in item.box)
            offset = (index + len(target) / 2) / len(value)
            point = (min_x + (max_x - min_x) * offset, item.center[1])
        self.x11.click(snapshot, point)
        self.pause()

    def wait_for(self, predicate, timeout=12, description="页面更新"):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.read()
            if predicate(last):
                return last
            time.sleep(0.15)
        self.save_snapshot(last, "timeout")
        raise ReservationError(f"等待{description}超时")

    def save_snapshot(self, snapshot, prefix="failure"):
        if snapshot is None:
            return
        self.log_dir.mkdir(mode=0o700, exist_ok=True)
        path = self.log_dir / f"{prefix}.png"
        snapshot.image.save(path)
        os.chmod(path, 0o600)
        LOG.info("已保存排障截图：%s", path)

    def already_succeeded(self):
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return False
        return state.get("status") == "success" and state.get("date") == self.target_date.isoformat()

    def record_success(self, seat):
        state = {
            "status": "success",
            "date": self.target_date.isoformat(),
            "seat": seat,
            "recorded_at": datetime.now(self.tz).isoformat(),
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, self.state_path)

    def select_building_date(self, snapshot):
        label = f"{self.target_date:%m-%d} 明天" if self.config.get("date_offset", 1) == 1 else f"{self.target_date:%m-%d}"
        self.click_text(snapshot, label, contains=True)

    def select_seat_date(self, snapshot):
        label = f"{self.target_date.isoformat()} 明天" if self.config.get("date_offset", 1) == 1 else self.target_date.isoformat()
        self.click_text(snapshot, label, contains=True)

    def configure_seat_page(self, snapshot=None):
        snapshot = snapshot or self.read()
        self.click_text(snapshot, "列表模式")
        self.select_seat_date(snapshot)
        self.click_text(snapshot, self.config["time_slot"], contains=True)
        snapshot = self.read()
        expected_date = self.target_date.isoformat()
        if expected_date not in snapshot.combined_text():
            raise ReservationError("页面日期与目标日期不一致，已停止操作")
        if normalize(self.config["time_slot"]) not in snapshot.combined_text():
            raise ReservationError("页面时段与配置不一致，已停止操作")
        if normalize(self.config["room"]) not in snapshot.combined_text():
            raise ReservationError("页面阅览室与配置不一致，已停止操作")
        return snapshot

    def enter_from_building(self, snapshot):
        self.select_building_date(snapshot)
        if not has_text(snapshot, self.config["floor"]):
            self.click_text(snapshot, self.config["building"])
            snapshot = self.wait_for(
                lambda current: has_text(current, self.config["floor"]),
                description="楼层列表",
            )
        self.click_text(snapshot, self.config["floor"])
        snapshot = self.wait_for(
            lambda current: has_text(current, "立即预约") and has_text(current, "选择阅览室", contains=True),
            description="阅览室列表",
        )
        room = normalize(self.config["room"])
        if room not in snapshot.combined_text():
            raise ReservationError(f"未找到配置的阅览室：{self.config['room']}")
        room_items = find_items(snapshot, self.config["room"], contains=True)
        buttons = find_items(snapshot, "立即预约")
        if room_items and buttons:
            room_x, room_y = room_items[0].center
            nearby = [
                button for button in buttons
                if button.center[1] > room_y and abs(button.center[0] - room_x) < 550
            ]
            button = min(
                nearby or buttons,
                key=lambda item: (item.center[0] - room_x) ** 2 + (item.center[1] - room_y) ** 2,
            )
            self.click_item(snapshot, button)
        else:
            self.click_text(snapshot, "立即预约")
        snapshot = self.wait_for(
            lambda current: has_text(current, "选择座位") and has_text(current, "地图模式"),
            timeout=18,
            description="选座页面",
        )
        return self.configure_seat_page(snapshot)

    def prepare(self):
        self.x11.activate(self.window_id)
        for _ in range(8):
            snapshot = self.read()
            if has_text(snapshot, "选择座位") and has_text(snapshot, "地图模式"):
                self.click_text(snapshot, "选择馆舍", contains=True)
                continue
            if has_text(snapshot, "选择阅览室", contains=True) and has_text(snapshot, "立即预约"):
                self.click_text(snapshot, "选择馆舍", contains=True)
                continue
            if (
                has_text(snapshot, "选择馆舍", contains=True)
                and has_text(snapshot, self.config["building"])
                and has_text(snapshot, "明天", contains=True)
            ):
                LOG.info("正在进入 %s%s%s", self.config["building"], self.config["floor"], self.config["room"])
                return self.enter_from_building(snapshot)
            if has_text(snapshot, "座位预约") and has_text(snapshot, "常用预约"):
                self.click_text(snapshot, "座位预约")
                continue
            LOG.info("当前页面不是已知预约页面，刷新后重试")
            self.x11.click(snapshot, (155, 42))
            time.sleep(1.5)
        raise ReservationError("无法进入选座页面，请检查微信登录状态和系统页面")

    def refresh_seat_page(self):
        snapshot = self.read()
        self.x11.click(snapshot, (155, 42))
        time.sleep(1.5)
        try:
            snapshot = self.wait_for(
                lambda current: (
                    has_text(current, "选择座位") and has_text(current, "地图模式")
                ) or (
                    has_text(current, "选择馆舍", contains=True)
                    and has_text(current, self.config["building"])
                    and has_text(current, "明天", contains=True)
                ),
                timeout=18,
                description="整点刷新",
            )
            if has_text(snapshot, "选择座位") and has_text(snapshot, "地图模式"):
                return self.configure_seat_page(snapshot)
            LOG.info("刷新后返回馆舍页，重新选择阅览室")
            return self.enter_from_building(snapshot)
        except ReservationError:
            LOG.warning("刷新后未进入已知预约页面，重新执行预约流程")
            return self.prepare()

    @staticmethod
    def available_seats(snapshot):
        region = (0, snapshot.state.height * 0.2, snapshot.state.width * 0.56, snapshot.state.height)
        seats = [
            item for item in snapshot.items
            if re.fullmatch(r"\d{3}", normalize(item.text)) and in_region(item, region)
        ]
        return sorted(seats, key=lambda item: (round(item.center[1] / 40), item.center[0]))

    @staticmethod
    def selected_seat(snapshot):
        labels = find_items(snapshot, "已选座位", contains=True)
        for label in labels:
            combined = re.search(r"已选座位:?(\d{3})", normalize(label.text))
            if combined:
                return combined.group(1)
            label_x, label_y = label.center
            for item in snapshot.items:
                value = normalize(item.text)
                x, y = item.center
                if re.fullmatch(r"\d{3}", value) and abs(y - label_y) < 55 and label_x - 20 < x < label_x + 450:
                    return value
        return None

    def select_seat(self, excluded):
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            snapshot = self.read()
            seats = self.available_seats(snapshot)
            by_number = {normalize(item.text): item for item in seats}
            chosen = None
            for number in self.config.get("seat_priority", []):
                number = str(number).zfill(3)
                if number not in excluded and number in by_number:
                    chosen = by_number[number]
                    break
            if chosen is None and self.config.get("fallback_any_available", True):
                chosen = next((item for item in seats if normalize(item.text) not in excluded), None)
            if chosen is not None:
                number = normalize(chosen.text)
                self.click_item(snapshot, chosen)
                current = self.wait_for(
                    lambda snap: self.selected_seat(snap) == number,
                    timeout=5,
                    description=f"选中座位 {number}",
                )
                LOG.info("已选择座位 %s（尚未提交）", number)
                return number, current
            time.sleep(0.4)
        raise SeatUnavailable("当前页面没有识别到可选座位")

    def submit(self):
        snapshot = self.read()
        if self.target_date.isoformat() not in snapshot.combined_text():
            raise ReservationError("提交前日期校验失败")
        self.click_text(snapshot, "立即预约")
        deadline = time.monotonic() + 18
        while time.monotonic() < deadline:
            snapshot = self.read()
            combined = snapshot.combined_text()
            if "预约成功" in combined or "成功预约" in combined:
                confirmations = find_items(snapshot, "确定") or find_items(snapshot, "知道了")
                if confirmations:
                    self.click_item(snapshot, confirmations[0])
                return
            if "已有预约" in combined or "已预约过" in combined:
                raise ReservationError("系统提示当前时段已有预约，请在我的中心核对")
            if any(word in combined for word in ("座位已被", "已被预约", "预约失败", "手慢", "不可预约", "已占用")):
                self.save_snapshot(snapshot, "seat-unavailable")
                confirmations = find_items(snapshot, "确定") or find_items(snapshot, "知道了")
                if confirmations:
                    self.click_item(snapshot, confirmations[0])
                raise SeatUnavailable("所选座位提交时不可用")
            dialog_visible = any(word in combined for word in ("确认预约", "预约确认", "确认信息", "是否确认"))
            if dialog_visible:
                for label in ("确定", "确认", "确认预约", "提交"):
                    confirmations = find_items(snapshot, label)
                    if confirmations:
                        self.click_item(snapshot, confirmations[0])
                        break
                else:
                    self.save_snapshot(snapshot, "unknown-confirmation")
                    raise ReservationError("出现未识别的确认对话框，未继续点击")
                continue
            time.sleep(0.2)
        self.save_snapshot(snapshot, "unknown-result")
        raise ReservationError("提交后未识别到明确结果，请人工核对，脚本不会重复提交")

    def wait_for_run_time(self):
        now = datetime.now(self.tz)
        run_time = datetime.strptime(self.config["run_time"], "%H:%M:%S").time()
        latest = datetime.strptime(self.config["latest_end_time"], "%H:%M:%S").time()
        target = datetime.combine(now.date(), run_time, self.tz)
        deadline = datetime.combine(now.date(), latest, self.tz)
        if now > deadline:
            raise ReservationError(f"已晚于允许执行时间 {latest}，本次不预约")
        target_epoch = target.timestamp()
        while True:
            remaining = target_epoch - time.time()
            if remaining <= 0:
                break
            time.sleep(min(1, remaining - 0.03) if remaining > 0.04 else 0.001)
        LOG.info("到达 %s，开始整点预约", self.config["run_time"])

    def run(self, now=False):
        if self.already_succeeded():
            LOG.info("%s 已有脚本成功记录，本次跳过", self.target_date)
            return "已有成功记录，跳过"
        self.prepare()
        if not now:
            self.wait_for_run_time()
        self.refresh_seat_page()
        attempted = set()
        max_attempts = int(self.config.get("max_seat_attempts", 3))
        for attempt in range(max_attempts):
            seat, _ = self.select_seat(attempted)
            attempted.add(seat)
            if self.dry_run:
                self.refresh_seat_page()
                self.wait_for(lambda current: self.selected_seat(current) is None, description="清空演练选座")
                return f"演练通过：{self.target_date} {self.config['room']} {seat}，未提交预约"
            try:
                self.submit()
                self.record_success(seat)
                return f"预约成功：{self.target_date} {self.config['room']} {seat}"
            except SeatUnavailable as error:
                LOG.warning("座位 %s 不可用：%s", seat, error)
                if attempt + 1 < max_attempts:
                    self.refresh_seat_page()
        raise ReservationError(f"连续 {max_attempts} 个座位均未预约成功")


def notify(summary, body):
    try:
        subprocess.run(["notify-send", summary, body], check=False, timeout=3)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ("timezone", "run_time", "latest_end_time", "building", "floor", "room", "time_slot")
    missing = [key for key in required if key not in config]
    if missing:
        raise ReservationError(f"配置缺少字段：{', '.join(missing)}")
    for key in ("run_time", "latest_end_time"):
        datetime.strptime(config[key], "%H:%M:%S")
    if not re.fullmatch(r"\d{2}:\d{2}\s*-\s*\d{2}:\d{2}", config["time_slot"]):
        raise ReservationError("time_slot 格式应为 HH:MM - HH:MM")
    if int(config.get("max_seat_attempts", 3)) < 1 or int(config.get("max_seat_attempts", 3)) > 5:
        raise ReservationError("max_seat_attempts 必须在 1 到 5 之间")
    return config


def main():
    parser = argparse.ArgumentParser(description="山东青年政治学院座位预约")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--now", action="store_true", help="立即执行，不等待中午十二点")
    parser.add_argument("--dry-run", action="store_true", help="只验证到选座，不提交预约")
    parser.add_argument("--check", action="store_true", help="只检查窗口和 OCR，不点击页面")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reservation = None
    try:
        config = load_config(args.config.resolve())
        reservation = Reservation(config, dry_run=args.dry_run or config.get("dry_run", False))
        if args.check:
            snapshot = reservation.read()
            print(
                f"检查通过：窗口 {snapshot.state.width}x{snapshot.state.height}，"
                f"识别到 {len(snapshot.items)} 段文字；目标日期 {reservation.target_date}"
            )
            return 0
        result = reservation.run(now=args.now)
        LOG.info(result)
        if not args.dry_run:
            notify("座位预约", result)
        print(result)
        return 0
    except Exception as error:
        LOG.exception("预约失败：%s", error)
        if reservation is not None:
            try:
                reservation.save_snapshot(reservation.read())
            except Exception:
                LOG.warning("无法保存失败截图", exc_info=True)
        notify("座位预约失败", str(error))
        print(f"预约失败：{error}", file=sys.stderr)
        return 1
    finally:
        if reservation is not None:
            reservation.close()


if __name__ == "__main__":
    raise SystemExit(main())

//手动进行无提交演练
cd /home/admistor/Html_Practice
SeatReservation/.venv/bin/python SeatReservation/reserve.py --dry-run --now
//查看运行日志
journalctl --user -u seat-reservation.service -n 50 --no-pager
//停用
systemctl --user disable --now seat-reservation.timer

import ctypes, json, signal, sys, time
class BudgetExpired(BaseException): pass
def alarm(*args): raise BudgetExpired()
lib = ctypes.PyDLL(sys.argv[1])
lib.busy_seconds.argtypes = [ctypes.c_double]
lib.busy_seconds.restype = None
signal.signal(signal.SIGALRM, alarm)
start = time.monotonic()
signal.setitimer(signal.ITIMER_REAL, 0.1)
print("ENTER_NATIVE", flush=True)
try:
    lib.busy_seconds(2.0)
except BudgetExpired:
    print(json.dumps({"alarm_budget_s":0.1,"handled_after_s":time.monotonic()-start}), flush=True)

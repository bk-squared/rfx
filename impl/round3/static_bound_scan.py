GiB = 2**30
def stage1(Nd, C):
    L = max(4, min(int(Nd*0.33), Nd-2)); M = Nd-L; k = min(M, L*C)
    return 1.87*16*(3*M*L*C + M*k + 2*k*L*C + 4*k*k), k, L
def static2(n_win, k, C, target):
    w = k + C
    size = min(n_win, max(w, int(target)//(16*w)))
    return 1.71*16*5*(size*w + w*w)
for target in (2**20, 8*2**20):
    print(f'host target {target/2**20:.0f} MiB')
    for C in (1, 2, 4, 8, 16, 64):
        worst = None   # largest static2 where static2 > stage1 (static check binding)
        default_hit = False
        for Nd in range(10, 40001):
            s1, k, L = stage1(Nd, C)
            s2 = static2(10**9, k, C, target)
            if s2 > s1 and (worst is None or s2 > worst[0]):
                worst = (s2, Nd, k, s1)
            if s1 <= 4*GiB < s2:
                default_hit = True
        print(f'  C={C:3d}: static stage 2 binding only below bound {worst[0]/2**20:8.2f} MiB (Nd={worst[1]}, k={worst[2]}, stage1 {worst[3]/2**20:.2f} MiB); refusal at 4 GiB that stage 1 admits: {default_hit}')
# the cavity fixture of test_we_refusal_keeps_cavity_main_s: Nd=480, L=158, M=322, C=2, k=316, n_win=3000
main = 1.87*16*(3*322*158*2 + 322*316 + 2*316*158*2 + 4*316**2)
long = 1.87*16*(3*490*240*2 + 490*480 + 2*480*240*2 + 4*480**2)
bound = (main+long)/2
for target in (2**20, 8*2**20):
    print(f'cavity fixture, target {target/2**20:.0f} MiB: stage1 {main/2**20:.2f} MiB, bound {bound/2**20:.2f} MiB, static2 {static2(3000, 316, 2, target)/2**20:.2f} MiB')
# memory-bound identify fixture: n_win 20000, Nd 178, C 1, k 58, actual rank 2
s1 = 1.87*16*(3*120*58 + 120*58 + 2*58*58 + 4*58**2)
print(f'identify fixture: stage1 {s1/2**20:.3f} MiB, static2(k=58) {static2(20000, 58, 1, 2**20)/2**20:.3f} MiB, '
      f'new actual stage2(r=2) {static2(20000, 2, 1, 2**20)/2**20:.3f} MiB, base stage2(r=2) {1.71*16*5*20000*2/2**20:.3f} MiB')

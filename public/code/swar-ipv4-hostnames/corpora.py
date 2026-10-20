# Generates c_shapes.txt, c_real32.txt, c_domains.txt, c_mixed.txt (2^20 lines each).
# Needs swift-endpoint's Tests/Resources/top-domains.csv in the current directory.
import random
random.seed(42)
doms=[l.strip().split(',')[-1] for l in open('top-domains.csv') if l.strip()][1:]
N=1<<20
def ip(): return '.'.join(str(random.randrange(256)) for _ in range(4))
shapes=[ip() for _ in range(N)]
pop=['127.0.0.1','192.168.1.1','10.0.0.1','8.8.8.8','1.1.1.1','192.168.0.1','0.0.0.0','255.255.255.255','8.8.4.4','1.0.0.1','172.16.0.1','10.0.0.138','192.168.1.254','208.67.222.222','9.9.9.9','169.254.169.254','10.10.10.1','192.168.100.1','100.64.0.1','224.0.0.251','239.255.255.250','192.0.2.1','198.51.100.1','203.0.113.1','64.233.160.0','151.101.1.69','140.82.112.3','104.16.132.229','13.107.42.14','17.253.144.10','31.13.71.36','52.84.150.39']
real32=[random.choice(pop) for _ in range(N)]
domains=[random.choice(doms) for _ in range(N)]
mixed=[random.choice(doms) if random.random()<0.9 else ip() for _ in range(N)]
for n,c in [('shapes',shapes),('real32',real32),('domains',domains),('mixed',mixed)]:
    open(f'c_{n}.txt','w').write('\n'.join(c)+'\n')

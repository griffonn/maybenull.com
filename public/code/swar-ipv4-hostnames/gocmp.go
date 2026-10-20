package main
import ("bufio";"fmt";"net";"net/netip";"os")
func main(){ s:=bufio.NewScanner(os.Stdin); for s.Scan(){ l:=s.Text()
 a,err:=netip.ParseAddr(l); x:="REJECT"; if err==nil { x=a.String() }
 ip:=net.ParseIP(l); y:="REJECT"; if ip!=nil { y=ip.String() }
 fmt.Printf("%-18q netip=%-16s net.ParseIP=%s\n",l,x,y) } }

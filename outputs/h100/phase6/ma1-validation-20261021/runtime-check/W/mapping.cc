#include <initializer_list>
#include "phase6_host.hpp"
int main() {
  for(int k: {512,2048}) for(int r=0;r<8;++r) for(int i=0;i<512;++i)
    if(phase6_host_lookup(16,32,k,r,i)!=((i%8+r)%8)*64+i/8)return 1;
  return phase6_host_lookup(16,32,1024,0,0)==-1 ? 0 : 2;
}

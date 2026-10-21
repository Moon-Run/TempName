#include <initializer_list>
#include "phase6_host.hpp"
int main() {
  for(int k: {512,2048}) for(int rank=0;rank<8;++rank)
    for(int i=0;i<512;++i)
      if(phase6_host_lookup(16,32,k,rank,i)!=((i%8+rank)%8)*64+i/8)return 1;
  if(phase6_host_lookup(16,32,1024,0,0)!=-1)return 2;
}

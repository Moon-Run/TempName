#pragma once
inline int phase6_host_lookup(int tm,int tn,int k,int rank,int i) {
 if(tm!=16 || tn!=32 || (k!=512 && k!=2048) || rank<0 || rank>=8 || i<0 || i>=512) return -1;
 return ((i%8+rank)%8)*64+i/8;
}

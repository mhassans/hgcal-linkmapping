#include <iostream>


void uvMapping(unsigned layer, std::pair<int,int> &uv) {
 unsigned sector(0);
 int offset;

 if(layer<=28) { // CE-E    
   if(uv.first>0 && uv.second>=0) return;

   offset=0;
   if(uv.first>=uv.second && uv.second<0) sector=2;
   else sector=1;

 } else if((layer%2)==1) { // CE-H Odd
   if(uv.first>=0 && uv.second>=0) return;

   offset=-1;    
   if(uv.first>uv.second && uv.second<0) sector=2;
   else sector=1;

 } else { // CE-H Even
   if(uv.first>=1 && uv.second>=1) return;

   offset=1;
   if(uv.first>=uv.second && uv.second<1) sector=2;
   else sector=1;
 }

 int up,vp;

 if(sector==1) {
   up=uv.second-uv.first;
   vp=-uv.first+offset;    

 } else {
   up=-uv.second+offset;
   vp=uv.first-uv.second+offset;
 }

 uv.first=up;
 uv.second=vp;
 return;
}

int main(int argc, char **argv){

  int u = atoi(argv[1]);
  int v = atoi(argv[2]);
  unsigned layer = atoi(argv[3]);
  std::pair<int,int> uv = std::make_pair(u,v);

  uvMapping(layer, uv);

  std::cout << "c++: " << uv.first << "," << uv.second << std::endl;
  
  return 0;
}

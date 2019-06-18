#include <Module.cpp>

using namespace std;

int foo(void)
{
  double c=1;
}

int main(int argc,char* argv[])
{
  //Run C components
  double a=1.0,b=2.0;
  double c=sum(a,b);
  printf("C: %g + %g  = %g\n",a,b,c);

  //Run C++ components
  Module *m=new Module();
  c=m->sumNumbers(a,b);
  printf("C++: %g + %g  = %g\n",a,b,c);
}

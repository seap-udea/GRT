#include <cstdio>
#include <cmath>

//Pure C routines
double sum(double a,double b)
{
  double c;
  return a+b;
}

double sub(double a,double b)
{
  double c=a-b;
  return c;
}

//Pure C++ routines
class Module
{
public:
  double sumNumbers(double a,double b)
  {
    return a+b;
  }
  double subNumbers(double a,double b)
  {
    return a-b;
  }
  double mulNumbers(double a,double b)
  {
    return a*b;
  }
};

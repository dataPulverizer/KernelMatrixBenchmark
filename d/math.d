import core.stdc.math: exp, expf, expl, fabs, fabsf, fabsl, log, logf, logl, 
      pow, powf, powl, sin, sinf, sinhl, sqrt, sqrtf, sqrtl, tanh, tanhf, tanhl;

/**
  Postfix for function names
*/
template Postfix(string type)
{
  static if(type == "float")
    enum Postfix = "f";
  else static if(type == "double")
    enum Postfix = "";
  else static if(type == "real")
    enum Postfix = "l";
}

/**
  Function representation
*/
template FunRepr(string fun, string type)
{
  enum FunRepr = 
  "T " ~ fun ~ "(T)(T x)
  if(is(T == " ~ type ~ "))
  {
    return " ~ fun ~ Postfix!(type) ~ "(x);
  }";
}

/**
  Generate the functions
*/
enum funcs = ["fabs", "exp", "log", "pow", "sin", "sqrt", "tanh"];
enum types = ["float", "double", "real"];

static foreach(fun; funcs)
    static foreach(type; types)
      mixin(FunRepr!(fun, type));

alias abs = fabs;

void demo()
{
  import std.stdio: writeln;
  writeln("Test float : ", exp(1.0f));
  writeln("Test double: ", exp(1.0));
  writeln("Test real  : ", exp(1.0L));
}

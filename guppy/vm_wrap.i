%module vm_wrap
#define __device__ 
%{
#include "bytecode.h"
#include "util.h"
%}

%exception {
  try {
    $function
  } catch (VMException& e) {
    PyErr_Format(PyExc_RuntimeError, "%s : at %s:%d", 
                 e.what.c_str(), e.file.c_str(), e.line);
    return NULL;
  }
}

%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>

class Instruction {
private:
  Instruction();
};

template <class T>
class InstructionT : public Instruction {
private:
};

%template() InstructionT< LoadVector >;
%template() InstructionT< LoadVector2 >;
%template() InstructionT< StoreVector >;
%template() InstructionT< Add >;
%template() InstructionT< IAdd >;
%template() InstructionT< Sub >;
%template() InstructionT< ISub >;
%template() InstructionT< Mul >;
%template() InstructionT< IMul >;
%template() InstructionT< Div >;
%template() InstructionT< IDiv >;
%template() InstructionT< MultiplyAdd >;
%template() InstructionT< IMultiplyAdd >;
%template() InstructionT< Map >;
%template() InstructionT< Map2 >;

%ignore eval_impl;

%include "dispatch_table.h"
%include "bytecode.h"
%include "config.h"

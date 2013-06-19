%module core

%include <std_string.i>
%include <std_map.i>
%include <std_vector.i>
%include <stdint.i>

%exception {
  try {
    $function
  } catch (VMException& e) {
    PyErr_Format(PyExc_RuntimeError, "%s : at %s:%d", 
                 e.what.c_str(), e.file.c_str(), e.line);
    return NULL;
  }
}

struct Instruction {
private:
    Instruction();
};

template<class SubClass>
struct InstructionT: public Instruction {
public:
  const uint16_t tag;
  const uint16_t size;
};

%template() InstructionT<LoadVector>;
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

%include "bytecode.h"

%{
#include "bytecode.h"
#include "util.h"
%}

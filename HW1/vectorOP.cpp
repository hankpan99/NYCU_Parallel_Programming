#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float result, cpValue;
  __pp_vec_float threshold = _pp_vset_float(9.999999f);
  __pp_vec_int remainExp;
  __pp_vec_int intZero = _pp_vset_int(0), intOne = _pp_vset_int(1);
  __pp_mask maskValid, maskPosExp, maskOutofRange;
  int reaminNum = N;

  for(int i = 0; i < N; i += VECTOR_WIDTH){
    // valid mask (prevent out of range)
    maskValid = _pp_init_ones(reaminNum);
    reaminNum -= VECTOR_WIDTH;

    // load data from memory & initialization
    _pp_vload_int(remainExp, exponents + i, maskValid);
    _pp_vload_float(cpValue, values + i, maskValid);
    _pp_vset_float(result, 1.0f, maskValid);

    // mask for exp > 0
    maskPosExp = _pp_init_ones(0);
    _pp_vlt_int(maskPosExp, intZero, remainExp, maskValid);
    
    // while exist exp > 0
    while(_pp_cntbits(maskPosExp)){
      _pp_vmult_float(result, result, cpValue, maskPosExp);
      
      // clamp value to 9.999
      maskOutofRange = _pp_init_ones(0);
      _pp_vlt_float(maskOutofRange, threshold, result, maskValid); // maskPosExp => maskValid (increase utilization) ???
      _pp_vmove_float(result, threshold, maskOutofRange);
      
      // exp -= 1
      _pp_vsub_int(remainExp, remainExp, intOne, maskPosExp);

      // mask for exp > 0
      _pp_vlt_int(maskPosExp, intZero, remainExp, maskValid);
    }

    // store result to memory
    _pp_vstore_float(output + i, result, maskValid); // maskPosExp => maskValid (increase utilization) ???
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  __pp_vec_float cpValue, result;
  __pp_mask maskAll = _pp_init_ones();
  int vecWidth;
  float sum = 0.0f;

  for(int i = 0; i < N; i += VECTOR_WIDTH){
    _pp_vload_float(cpValue, values + i, maskAll);

    vecWidth = VECTOR_WIDTH;

    while(vecWidth > 1){
      // add up adjacent pairs
      _pp_hadd_float(cpValue, cpValue);
      
      // even-odd interleaving
      _pp_interleave_float(result, cpValue);
      _pp_vmove_float(cpValue, result, maskAll);
      
      vecWidth /= 2;
    }

    sum += result.value[0];
  }

  return sum;
}
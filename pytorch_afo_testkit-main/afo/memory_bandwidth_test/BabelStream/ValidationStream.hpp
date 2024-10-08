#pragma once

#include <iostream>
#include <stdexcept>

#include "Stream.h"

template <class T>
class ValidationStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;
    T a, b, c;

  public:
    ValidationStream(int array_size) {
        this->array_size = array_size;
    }
    ~ValidationStream() {}

    virtual float read() override {
        return 0.0f;
    }
    virtual float write() override {
        c = startC;
        return 0.0f;
    }
    virtual float copy() override {
        c = a;
        return 0.0f;
    }
    virtual float add() override {
        c = a + b;
        return 0.0f;
    }
    virtual float mul() override {
        b = startScalar * c;
        return 0.0f;
    }
    virtual float triad() override {
        a = b + startScalar * c;
        return 0.0f;
    }
    virtual T dot() override {
        return a * b * array_size;
    }

    virtual void init_arrays(T initA, T initB, T initC) override {
          this->a = initA;
          this->b = initB;
          this->c = initC;
    }
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override {
        std::fill(a.begin(), a.end(), this->a);
        std::fill(b.begin(), b.end(), this->b);
        std::fill(c.begin(), c.end(), this->c);
    }

};

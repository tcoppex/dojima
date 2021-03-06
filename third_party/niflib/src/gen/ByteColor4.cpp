/* Copyright (c) 2006, NIF File Format Library and Tools
All rights reserved.  Please see niflib.h for license. */

//---THIS FILE WAS AUTOMATICALLY GENERATED.  DO NOT EDIT---//

//To change this file, alter the niftools/docsys/gen_niflib.py Python script.

#include "nif/gen/ByteColor4.h"
using namespace Niflib;

//Constructor
ByteColor4::ByteColor4() : r((ubyte_t)0), g((ubyte_t)0), b((ubyte_t)0), a((ubyte_t)0) {};

//Copy Constructor
ByteColor4::ByteColor4( const ByteColor4 & src ) {
	*this = src;
};

//Copy Operator
ByteColor4 & ByteColor4::operator=( const ByteColor4 & src ) {
	this->r = src.r;
	this->g = src.g;
	this->b = src.b;
	this->a = src.a;
	return *this;
};

//Destructor
ByteColor4::~ByteColor4() {};

//--BEGIN MISC CUSTOM CODE--//
//--END CUSTOM CODE--//

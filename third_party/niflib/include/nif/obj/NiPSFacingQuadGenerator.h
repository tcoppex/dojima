/* Copyright (c) 2006, NIF File Format Library and Tools
All rights reserved.  Please see niflib.h for license. */

//-----------------------------------NOTICE----------------------------------//
// Some of this file is automatically filled in by a Python script.  Only    //
// add custom code in the designated areas or it will be overwritten during  //
// the next update.                                                          //
//-----------------------------------NOTICE----------------------------------//

#ifndef _NIPSFACINGQUADGENERATOR_H_
#define _NIPSFACINGQUADGENERATOR_H_

//--BEGIN FILE HEAD CUSTOM CODE--//

//--END CUSTOM CODE--//

#include "nif/obj/NiObject.h"
namespace Niflib {

class NiPSFacingQuadGenerator;
typedef Ref<NiPSFacingQuadGenerator> NiPSFacingQuadGeneratorRef;

/*!  */
class NiPSFacingQuadGenerator : public NiObject {
public:
	/*! Constructor */
	NIFLIB_API NiPSFacingQuadGenerator();

	/*! Destructor */
	NIFLIB_API virtual ~NiPSFacingQuadGenerator();

	/*!
	 * A constant value which uniquly identifies objects of this type.
	 */
	NIFLIB_API static const Type TYPE;

	/*!
	 * A factory function used during file reading to create an instance of this type of object.
	 * \return A pointer to a newly allocated instance of this type of object.
	 */
	NIFLIB_API static NiObject * Create();

	/*!
	 * Summarizes the information contained in this object in English.
	 * \param[in] verbose Determines whether or not detailed information about large areas of data will be printed out.
	 * \return A string containing a summary of the information within the object in English.  This is the function that Niflyze calls to generate its analysis, so the output is the same.
	 */
	NIFLIB_API virtual string asString( bool verbose = false ) const;

	/*!
	 * Used to determine the type of a particular instance of this object.
	 * \return The type constant for the actual type of the object.
	 */
	NIFLIB_API virtual const Type & GetType() const;

	//--BEGIN MISC CUSTOM CODE--//

	//--END CUSTOM CODE--//
protected:
	/*! Unknown. */
	ubyte_t unknown1;
	/*! Unknown. */
	ubyte_t unknown2;
	/*! Unknown. */
	ubyte_t unknown3;
	/*! Unknown. */
	ubyte_t unknown4;
	/*! Unknown. */
	ubyte_t unknown5;
	/*! Unknown. */
	ubyte_t unknown6;
	/*! Unknown. */
	ubyte_t unknown7;
	/*! Unknown. */
	ubyte_t unknown8;
	/*! Unknown. */
	ubyte_t unknown9;
	/*! Unknown. */
	ubyte_t unknown10;
	/*! Unknown. */
	ubyte_t unknown11;
	/*! Unknown. */
	ubyte_t unknown12;
public:
	/*! NIFLIB_HIDDEN function.  For internal use only. */
	NIFLIB_HIDDEN virtual void Read( istream& in, list<unsigned int> & link_stack, const NifInfo & info );
	/*! NIFLIB_HIDDEN function.  For internal use only. */
	NIFLIB_HIDDEN virtual void Write( ostream& out, const map<NiObjectRef,unsigned int> & link_map, list<NiObject *> & missing_link_stack, const NifInfo & info ) const;
	/*! NIFLIB_HIDDEN function.  For internal use only. */
	NIFLIB_HIDDEN virtual void FixLinks( const map<unsigned int,NiObjectRef> & objects, list<unsigned int> & link_stack, list<NiObjectRef> & missing_link_stack, const NifInfo & info );
	/*! NIFLIB_HIDDEN function.  For internal use only. */
	NIFLIB_HIDDEN virtual list<NiObjectRef> GetRefs() const;
	/*! NIFLIB_HIDDEN function.  For internal use only. */
	NIFLIB_HIDDEN virtual list<NiObject *> GetPtrs() const;
};

//--BEGIN FILE FOOT CUSTOM CODE--//

//--END CUSTOM CODE--//

} //End Niflib namespace
#endif

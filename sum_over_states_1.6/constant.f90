

!-------- to be costumized by user (usually done in the makefile)-------
!#define vector              compile for vector machine
!#define essl                use ESSL instead of LAPACK
!#define single_BLAS         use single prec. BLAS

!#define wNGXhalf            gamma only wavefunctions (X-red)
!#define wNGZhalf            gamma only wavefunctions (Z-red)

!#define 1             charge stored in REAL array (X-red)
!#define NGZhalf             charge stored in REAL array (Z-red)
!#define NOZTRMM             replace ZTRMM by ZGEMM
!#define REAL_to_DBLE        convert REAL() to DBLE()
!#define MPI                 compile for parallel machine with MPI
!------------- end of user part         --------------------------------
!
!   charge density: half grid mode X direction
!
!
!   charge density real
!
!
!   wavefunctions: full grid mode
!
!
!   wavefunctions complex
!
!
!   common definitions
!





!************************************************************************
! RCS:  $Id: constant.F,v 1.1 2000/11/15 08:13:54 kresse Exp $
!
!  this module contains some control data structures for VASP
!
!***********************************************************************
      MODULE CONSTANT
      USE prec
      INCLUDE "constant.inc"
      CONTAINS

      SUBROUTINE NUP_CONST()
      END SUBROUTINE

      END MODULE

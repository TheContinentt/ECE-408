
#ifndef INTERRUPT_DESC_TABLE_H
#define INTERRUPT_DESC_TABLE_H

#include "i8259.h"
#include "keyboard.h"
#include "RTC.h"
#include "pit.h"
#include "mouse.h"
#include "x86_desc.h"
#include "syscall.h"
#include "system_call_handler.h"
#include "lib.h"

/*
	Reference: wiki.osdev.org IDT section
*/

/*
 *   DESCRIPTION: 20 exception handlers we handled
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: print corresponding exception line, halt program
 */
 
 /* prototype of div_by_zero_error_expt.*/
void div_by_zero_error_expt ();

 /* prototype of debug_expt.*/
void debug_expt();

 /* prototype of NMI_expt.*/
void NMI_expt();

 /* prototype of breakpoint_expt.*/
void breakpoint_expt();

 /* prototype of overflow_expt.*/
void overflow_expt();

 /* prototype of BR_exceed_expt.*/
void BR_exceed_expt();

 /* prototype of invalid_op_expt.*/
void invalid_op_expt();

 /* prototype of dev_unavailable_expt.*/
void dev_unavailable_expt();

 /* prototype of double_fault_expt.*/
void double_fault_expt();

 /* prototype of cop_seg_expt.*/
void cop_seg_expt();

 /* prototype of invalid_tss_expt.*/
void invalid_tss_expt();

 /* prototype of seg_not_present_expt.*/
void seg_not_present_expt();

 /* prototype of stack_seg_fault_expt.*/
void stack_seg_fault_expt();

 /* prototype of ge_prot_fault_expt.*/
void ge_prot_fault_expt();

 /* prototype of page_fault_expt.*/
void page_fault_expt();

 /* prototype of float_ptr_expt.*/
void float_ptr_expt();

 /* prototype of align_expt.*/
void align_expt();

 /* prototype of machine_expt.*/
void machine_expt();

 /* prototype of SIMD_expt.*/
void SIMD_expt();

 /* prototype of virt_expt.*/
void virt_expt();

 /* prototype of security_expt.*/
void security_expt();

/*Initialzie the IDT*/
void idt_init();

#endif



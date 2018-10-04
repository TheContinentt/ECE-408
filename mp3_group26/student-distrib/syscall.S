# the label that stores the eax.
.data
store_eax: .long 0x0
TWELVE = 12
ONE = 1
TEN = 10
# list of system calls and its number.
# * syscall (linkage):
# * Description: link to the differnt system calss by the jumptable
# * input: eax
# * output: none
# * return value:none
# * side effect: link to differnt system calls
# */
.text
system_calls:
	.long 	0
	.long 	halt
	.long 	execute
	.long 	read
	.long 	write
	.long 	open
	.long 	close
	.long	getargs
	.long 	vidmap
	.long 	set_handler
	.long 	sigreturn

# mark syscall_handler as global.
.global syscall_handler

syscall_handler:
	#first check value
	cmpl	$ONE, %eax
	jl		invalid_syscall
	cmpl	$TEN, %eax
	jg		invalid_syscall
	
	#save all registers
	pushl	%ecx
	pushl	%edx
	pushl	%ebx
	pushl	%esi
	pushl	%edi
	pushl	%ebp
	#push three arguments 
	pushl	%edx
	pushl	%ecx
	pushl	%ebx
	
	#jump table to C functions	
	call	*system_calls(,%eax, 4)
	
	# store eax to label to prevent potential eax change.
	movl	%eax, store_eax
	
	#tear down arguments 
	addl	$TWELVE, %esp
	#restore all registers
	popl	%ebp
	popl	%edi
	popl	%esi
	popl	%ebx
	popl	%edx
	popl	%ecx
	
	# restore the eax value from the label.
	movl	store_eax, %eax
	iret
	
# is is invalid syscall, return -1 by setting eax = -1.
invalid_syscall:
	movl	$-1, %eax
	iret

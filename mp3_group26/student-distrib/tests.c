#include "tests.h"
#include "x86_desc.h"
#include "lib.h"
#include "system_call_handler.h"
#include "RTC.h"
#include "file_system.h"
#define PASS 1
#define FAIL 0

/* format these macros as you see fit */
#define TEST_HEADER 	\
	printf("[TEST %s] Running %s at %s:%d\n", __FUNCTION__, __FUNCTION__, __FILE__, __LINE__)
#define TEST_OUTPUT(name, result)	\
	printf("[TEST %s] Result = %s\n", name, (result) ? "PASS" : "FAIL");

static inline void assertion_failure(){
	/* Use exception #15 for assertions, otherwise
	   reserved by Intel */
	asm volatile("int $15");
}


/* Checkpoint 1 tests */

/* IDT Test - Example
 *
 * Asserts that first 10 IDT entries are not NULL
 * Inputs: None
 * Outputs: PASS/FAIL
 * Side Effects: None
 * Coverage: Load IDT, IDT definition
 * Files: x86_desc.h/S
 */
int idt_test(){
	TEST_HEADER;

	int i;
	int result = PASS;
	for (i = 0; i < 10; ++i){
		if ((idt[i].offset_15_00 == NULL) &&
			(idt[i].offset_31_16 == NULL)){
			assertion_failure();
			result = FAIL;
		}
	}

	return result;
}




/* exception_test
 * divide by zero to test the first entry of IDT
 * Inputs: None
 * Outputs: None
 * Side Effects: send divide by zero exception and halt program
 */

void exception_test() {
	int k;
	int a = 0;
	k = k/a;
}
/* page_test_0
 * test the first in the page virtual memory by dereferencing it
 * Inputs: None
 * Outputs: None
 * Side Effects: page fault reported, printed, halt program
 */
void page_test_0() {
	int* temp;
	temp = 0;
	int a;
	a = (int)*temp;
}
/* page_test_C0000
 * test the address in the page virtual memory below video memory by dereferencing it
 * Inputs: None
 * Outputs: None
 * Side Effects: page fault reported, printed, halt program
 */
void page_test_C0000() {
	int* temp;
	temp = (int *)0xC0000;
	int a;
	a = (int)*temp;
}
/* page_test_500000
 * test the address in the page virtual memory in the kernel space by dereferencing it
 * Inputs: None
 * Outputs: None
 * Side Effects: no exception happens
 */
void page_test_kernel() {
	int* temp;
	temp = (int *)0x500000;
	int a;
	a = (int)*temp;
}
/* page_test_900000
 * test the address in the page virtual memory out of kernel space range by dereferencing it
 * Inputs: None
 * Outputs: None
 * Side Effects: page fault reported, printed, halt program
 */
void page_test_outkernel() {
	int* temp;
	temp = (int *)0x900000;
	int a;
	a = (int)*temp;
}

/* Checkpoint 2 tests */
/* RTC_open_test
 * Discription: test whether RTC_open works fine.
 * Inputs: None
 * Outputs: '1' are put on the screen periodically.
 * Side Effects: '1' is put on the screen in a specific frequency if RTC_open works.
 */
void RTC_open_test(){
		RTC_open(0);
		while(1){
			RTC_read(0,0,0);
			putc('1');
		}
}

/* RTC_write_test
 * Discription: test whether RTC_write works fine.
 * Inputs: freq: the freq that we want to set to the RTC.
 * Outputs: '1' are put on the screen periodically.
 * Side Effects: none.
 */
void RTC_write_test(int freq){
		RTC_write(0,(int*)&freq,4);
		while(1){
			RTC_read(0,0,0);
			putc('1');
		}
}

/* RTC_write_test
 * Discription: test whether RTC_close works fine.
 * Inputs: none.
 * Outputs: print "rtc close test success" if passes test.
 * Side Effects: none.
 */
void RTC_close_test(){
	if(!RTC_close(0))
		printf("rtc close test success\n");
}

/* terminal_open_test
 * Discription: test whether terminal_open works fine.
 * Inputs: none.
 * Outputs: print "terminal open test success" if passes test.
 * Side Effects: none.
 */
void terminal_open_test(){
	if(!terminal_open(0))
		printf("terminal open test success\n");
}

/* terminal_close_test
 * Discription: test whether terminal_close works fine.
 * Inputs: none.
 * Outputs: print "terminal close test success" if passes test.
 * Side Effects: none.
 */
void terminal_close_test(){
	if(!terminal_close(0))
		printf("terminal close test success\n");
}

/* terminal_read_write_test
 * Discription: test whether terminal_read and terminal_write work fine.
 * Inputs: none.
 * Outputs: print what we input on the keyboard to terminal twice.
 * Side Effects: none.
 */
void terminal_read_write_test() {
	uint8_t buf[130];
	terminal_read(0, buf, 130);
	terminal_write(0, buf, 130);
}

/* file_read_test_frame1
 * Discription: test whether file_read works fine.
 * Inputs: none.
 * Outputs: print the file content to terminal if read succeessfully.
 *					Otherwise, print "open failed".
 * Side Effects: none.
 */
void file_read_test_frame1() {
	int8_t* filename;
	int i = 0;
	filename = "frame1.txt";
	if(file_open((uint8_t*)filename)==-1){
		printf("open failed\n");
		return;}
	uint8_t buf[200];
	int len;
	len = file_read(0,buf,200);
	for(i = 0; i<len ; i ++)
		putc(buf[i]);
}

/* file_read_test_frame0
 * Discription: test whether file_read works fine.
 * Inputs: none.
 * Outputs: print the file content to terminal if read succeessfully.
 *					Otherwise, print "open failed".
 * Side Effects: none.
 */
void file_read_test_frame0() {
	int8_t* filename;
	int i = 0;
	filename = "frame0.txt";
	if(file_open((uint8_t*)filename)==-1){
		printf("open failed\n");
		return;}
	uint8_t buf[200];
	int len;
	len = file_read(0,buf,200);
	for(i = 0; i<len ; i ++)
		putc(buf[i]);
}

/* file_read_test_large
 * Discription: test whether file_read works fine.
 * Inputs: none.
 * Outputs: print the file content to terminal if read succeessfully.
 *					Otherwise, print "open failed".
 * Side Effects: none.
 */
void file_read_test_large() {
	int8_t* filename;
	int i = 0;
	filename = "verylargetextwithverylongname.txt";
	if(file_open((uint8_t*)filename)==-1){
		printf("open failed\n");
		return;}
	uint8_t buf[4096];
	int len;
	len = file_read(0,buf,4096);
	for(i = 0; i<len ; i ++)
		putc(buf[i]);
}

/* file_read_test_execute
 * Discription: test whether file_read works fine.
 * Inputs: none.
 * Outputs: print the file content to terminal if read succeessfully.
 *					Otherwise, print "open failed".
 * Side Effects: none.
 */
void file_read_test_execute() {
	int8_t* filename;
	int i = 0;
	filename = "cat";
	if(file_open((uint8_t*)filename)==-1){
		printf("open failed\n");
		return;}
	uint8_t buf[5445];
	int len;
	len = file_read(0,buf,5445);
	for(i = 0; i<len ; i ++)
		putc(buf[i]);
}

/* file_close_test
 * Discription: test whether file_close works fine.
 * Inputs: none.
 * Outputs: print "file close test success" if works succeessfully.
 * Side Effects: none.
 */
void file_close_test(){
	if(!file_close(0))
		printf("file close test success\n");
}

/* file_write_test
 * Discription: test whether file_write works fine.
 * Inputs: none.
 * Outputs: print "file write test success" if works succeessfully.
 * Side Effects: none.
 */
void file_write_test(){
	if(file_write(0,(void*)0,0)==-1)
		printf("file write test success\n");
}

/* dir_write_test
 * Discription: test whether dir_write works fine.
 * Inputs: none.
 * Outputs: print "dir write test success" if works succeessfully.
 * Side Effects: none.
 */
void dir_write_test(){
	if(dir_write(0,(void*)0,0)==-1)
		printf("dir write test success\n");
}

/* dir_read_test
 * Discription: test whether dir_read works fine.
 * Inputs: none.
 * Outputs: print directory data to terminal if works succeessfully.
 * Side Effects: none.
 */
void dir_read_test() {
	uint8_t buf[32];
	int len;
	while ((len=dir_read(0, buf, 32)) != -1) {
		terminal_write(0, buf, len);
		putc('\n');
	}
}

/* dir_open_test
 * Discription: test whether dir_open works fine.
 * Inputs: none.
 * Outputs: print "dir open test success" to terminal if works succeessfully.
 * Side Effects: none.
 */
void dir_open_test(){
	int8_t* a;
	a= ".";
	if(dir_open((uint8_t*)a)!=-1)
		printf("dir open test success\n");
}

/* dir_close_test
 * Discription: test whether dir_close works fine.
 * Inputs: none.
 * Outputs: print "dir close test success" to terminal if works succeessfully.
 * Side Effects: none.
 */
void dir_close_test(){
	if(!dir_close(0))
		printf("dir close test success\n");
}
/* Checkpoint 3 tests */
/* Checkpoint 4 tests */
/* Checkpoint 5 tests */


/* Test suite entry point */
void launch_tests(){
	TEST_OUTPUT("idt_test", idt_test());
//RTC TEST

	//RTC_open_test();
	//RTC_write_test(256);
//	RTC_close_test();
//terminal TEST
	//terminal_open_test();
	//terminal_close_test();
//	printf("length: %d\n", terminal_write(0, "12345", 5));
	//printf("length: %d\n", terminal_write(0, "12345", -1));
	//printf("length: %d\n", terminal_write(0, "12345", 4));
//	printf("length: %d\n", terminal_write(0, "12345", 6));
//	terminal_read_write_test();


//file_system TEST
//	dir_open_test();
//	dir_close_test();
//	dir_write_test();
//	file_close_test();
//	file_write_test();

//syscall TEST

/*
asm(
	"PUSHL	%EBX          "
	"MOVL	$number,%EAX  "
	"MOVL	8(%ESP),%EBX  "
	"MOVL	12(%ESP),%ECX "
	"MOVL	16(%ESP),%EDX "
	"INT	$0x80         "
	"POPL	%EBX          "
	"RET"

uint8_t buf[3];
	if(-1==read(0,NULL,2))
		printf("read invalid buf test: PASS\n");
	if(-1==read(8,buf,2))
		printf("read invalid fd(out of bound) test: PASS\n");
	if(-1==read(5,buf,2))
		printf("read invalid fd(not open) test: PASS\n");
	if(4==read(0,buf,4))
		printf("read overflow test: PASS\n");
	if(2==read(0,buf,2))
		printf("read overflow test1: PASS\n");*/
	
//	file_read_test_frame0();
	//file_read_test_frame1();
	//file_read_test_large();
	//file_read_test_execute();

}

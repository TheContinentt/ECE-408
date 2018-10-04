/* lib.h - Defines for useful library functions
 * vim:ts=4 noexpandtab
 */

#ifndef _LIB_H
#define _LIB_H
#define NUM_COLS    80
#define NUM_ROWS    25
#define MASK_FF				0xFF
#define MASK_F				0xF
#define MASK_E				0xE
#define MASK_8				0x8
#define VIDEO_MEM				0xB8000
#define VIDEO_WIDTH				80
#define BOT_LINE				3840
#define FONT_COLOR				10
#define BACKGROND_COLOR_OFFSET	16
#define ASCII_ZERO				48
#define ASCII_PERCENT			37
#define LOADING_TIME			2
#define LOADING_SCALE			8
#define CURSUR_BASE				13
#define CURSUR_PORTA			0x3D4
#define CURSUR_PORTB			0x3D5
#define CURSUR_PORTC			0x3E0
#define NUM_A					0xA
#define NUM_B					0xB
#define NUM_C					0xC0
#define NUM_E					0xE0
#define NUM_20					0x20
#define TEN						10
#define BUFFER_SIZE				20
#define ONE_KB 1024
#define FOUR_KB 4096
#include "types.h"

int32_t printf(int8_t *format, ...);

/* this function put the input character to screen.*/
void putc(uint8_t c);
void putc_background(uint8_t c);
void putc_testa(uint8_t c);
int32_t puts(int8_t *s);
int8_t *itoa(uint32_t value, int8_t* buf, int32_t radix);
int8_t *strrev(int8_t* s);
uint32_t strlen(const int8_t* s);
void clear(void);

void* memset(void* s, int32_t c, uint32_t n);
void* memset_word(void* s, int32_t c, uint32_t n);
void* memset_dword(void* s, int32_t c, uint32_t n);
void* memcpy(void* dest, const void* src, uint32_t n);
void* memmove(void* dest, const void* src, uint32_t n);
int32_t strncmp(const int8_t* s1, const int8_t* s2, uint32_t n);
int8_t* strcpy(int8_t* dest, const int8_t*src);
int8_t* strncpy(int8_t* dest, const int8_t*src, uint32_t n);

/* Userspace address-check functions */
int32_t bad_userspace_addr(const void* addr, int32_t len);
int32_t safe_strncpy(int8_t* dest, const int8_t* src, int32_t n);

void test_interrupts(void);


/* struct to keep track information of each terminal */
typedef struct terminal_t {
	char share_buf[128];
	char typed_buf[128];
	int typing_finish;
	int screen_x;
    int screen_y;
	int vid_on;
	//int shell_esp0;
	//int shell_cr3;
	//int shell_esp;
	//int shell_ebp;
	int	phys_mem_idx[6];
	char* video_mem;
	int curr_char_idx;
	char buffer_for_ec[BUFFER_SIZE][128];
	int buffer_ec_idx;
	int buffer_size;
	int hello_flag;
}terminal_t;  //check !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


/* array to keep the status of three terminals */
int32_t phys_addr_available[6];
terminal_t terminals[3];
int global_esp;
int global_ebp;
int	curr_terminal_idx;
int curr_running_terminal_idx;
int first_rount_flag;
int order[3];
int temp_curr_running_terminal_idx;
volatile int loading_counter;
int flag_first_execute;
int wait;

int pde_0[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pte_zero[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pte_one[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_1[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_2[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_3[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_4[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_5[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int pde_6[ONE_KB] __attribute__ ((aligned (FOUR_KB)));
int* pde_arr[7];
/* struct of file description table.*/
typedef struct file_operation_table_t {
	int32_t (*close)(int32_t fd);
	int32_t (*open)(const uint8_t* filename);
	int32_t (*write)(int32_t fd, const void* buf, int32_t nbyte);
	int32_t (*read)(int32_t fd, void* buf, int32_t nbytes);
}file_operation_table_t;


/* the structure of one entry of the file descriptor.*/
typedef struct file_descriptor_t {
	file_operation_table_t* file_operation_table_ptr;
	uint32_t inode;
	uint32_t file_position;
	uint32_t flags;
}file_descriptor_t;


/* Port read functions */
/* Inb reads a byte and returns its value as a zero-extended 32-bit
 * unsigned int */
static inline uint32_t inb(port) {
    uint32_t val;
    asm volatile ("             \n\
            xorl %0, %0         \n\
            inb  (%w1), %b0     \n\
            "
            : "=a"(val)
            : "d"(port)
            : "memory"
    );
    return val;
}

/* Reads two bytes from two consecutive ports, starting at "port",
 * concatenates them little-endian style, and returns them zero-extended
 * */
static inline uint32_t inw(port) {
    uint32_t val;
    asm volatile ("             \n\
            xorl %0, %0         \n\
            inw  (%w1), %w0     \n\
            "
            : "=a"(val)
            : "d"(port)
            : "memory"
    );
    return val;
}

/* Reads four bytes from four consecutive ports, starting at "port",
 * concatenates them little-endian style, and returns them */
static inline uint32_t inl(port) {
    uint32_t val;
    asm volatile ("inl (%w1), %0"
            : "=a"(val)
            : "d"(port)
            : "memory"
    );
    return val;
}

#define REP_OUTSB(port, source, count)                  \
do {                                                    \
    asm volatile("                                    \n\
        1: movb 0(%1), %%al                           \n\
        outb %%al, (%w2)                              \n\
        incl %1                                       \n\
        decl %0                                       \n\
        jne 1b                                        \n\
        "                                               \
        : /* no outputs */                              \
        : "c"((count)), "S"((source)), "d"((port))      \
        : "eax", "memory", "cc"                         \
    );                                                  \
} while (0)
/* macro used to write two bytes to two consecutive ports */
#define OUTW(port, val)                                 \
do {                                                    \
    asm volatile("                                    \n\
        outw %w1, (%w0)                               \n\
        "                                               \
        : /* no outputs */                              \
        : "d"((port)), "a"((val))                       \
        : "memory", "cc"                                \
    );                                                  \
} while (0)

/*
 * macro used to write an array of two-byte values to two consecutive ports
 */
#define REP_OUTSW(port, source, count)                  \
do {                                                    \
    asm volatile("                                    \n\
        1: movw 0(%1), %%ax                           \n\
        outw %%ax, (%w2)                              \n\
        addl $2, %1                                   \n\
        decl %0                                       \n\
        jne 1b                                        \n\
        "                                               \
        : /* no outputs */                              \
        : "c"((count)), "S"((source)), "d"((port))      \
        : "eax", "memory", "cc"                         \
    );                                                  \
} while (0)

/* Writes a byte to a port */
#define outb(data, port)                \
do {                                    \
    asm volatile ("outb %b1, (%w0)"     \
            :                           \
            : "d"(port), "a"(data)      \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Writes two bytes to two consecutive ports */
#define outw(data, port)                \
do {                                    \
    asm volatile ("outw %w1, (%w0)"     \
            :                           \
            : "d"(port), "a"(data)      \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Writes four bytes to four consecutive ports */
#define outl(data, port)                \
do {                                    \
    asm volatile ("outl %l1, (%w0)"     \
            :                           \
            : "d"(port), "a"(data)      \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Clear interrupt flag - disables interrupts on this processor */
#define cli()                           \
do {                                    \
    asm volatile ("cli"                 \
            :                           \
            :                           \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Save flags and then clear interrupt flag
 * Saves the EFLAGS register into the variable "flags", and then
 * disables interrupts on this processor */
#define cli_and_save(flags)             \
do {                                    \
    asm volatile ("                   \n\
            pushfl                    \n\
            popl %0                   \n\
            cli                       \n\
            "                           \
            : "=r"(flags)               \
            :                           \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Set interrupt flag - enable interrupts on this processor */
#define sti()                           \
do {                                    \
    asm volatile ("sti"                 \
            :                           \
            :                           \
            : "memory", "cc"            \
    );                                  \
} while (0)

/* Restore flags
 * Puts the value in "flags" into the EFLAGS register.  Most often used
 * after a cli_and_save_flags(flags) */
#define restore_flags(flags)            \
do {                                    \
    asm volatile ("                   \n\
            pushl %0                  \n\
            popfl                     \n\
            "                           \
            :                           \
            : "r"(flags)                \
            : "memory", "cc"            \
    );                                  \
} while (0)

#endif /* _LIB_H */

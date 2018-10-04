#include "modex.h"
#include "text.h"
#include "lib.h"
/* Mode X and general VGA parameters */
static unsigned char* mem_image = (unsigned char*)0xA0000;    /* pointer to start of video memory */
static unsigned short target_img;   /* offset of displayed screen image */
static void VGA_blank(int blank_bit);
static void set_seq_regs_and_reset(unsigned short table[NUM_SEQUENCER_REGS], unsigned char val);
static void set_CRTC_registers(unsigned short table[NUM_CRTC_REGS]);
static void set_attr_registers(unsigned char table[NUM_ATTR_REGS * 2]);
static void set_graphics_registers(unsigned short table[NUM_GRAPHICS_REGS]);
static void fill_palette_text();
static void write_font_data();
static void fill_palette_mode_x();
/*
 * macro used to target a specific video plane or planes when writing
 * to video memory in mode X; bits 8-11 in the mask_hi_bits enable writes
 * to planes 0-3, respectively
 */
#define SET_WRITE_MASK(mask_hi_bits)                    \
do {                                                    \
    asm volatile("                                    \n\
        movw $0x03C4, %%dx  /* set write mask */      \n\
        movb $0x02, %b0                               \n\
        outw %w0, (%%dx)                              \n\
        "                                               \
        :                                               \
        : "a"((mask_hi_bits))                           \
        : "edx", "memory"                               \
    );                                                  \
} while (0)
/* VGA register settings for mode X */
static unsigned short mode_X_seq[NUM_SEQUENCER_REGS] = {
    0x0100, 0x2101, 0x0F02, 0x0003, 0x0604
};
static unsigned short mode_X_CRTC[NUM_CRTC_REGS] = {
    0x5F00, 0x4F01, 0x5002, 0x8203, 0x5404, 0x8005, 0xBF06, 0x1F07,
    0x0008, 0x0109, 0x000A, 0x000B, 0x000C, 0x000D, 0x000E, 0x000F,
    0x9C10, 0x8E11, 0x8F12, 0x2813, 0x0014, 0x9615, 0xB916, 0xE317,
    0x8018
};//change the line comparater register and 8th bit in overflow register to represent 2*(200-18). 18 is height of the bar, 200 is the height of the logical window
//times 2 because each pixel counts as two line operation 
static unsigned char mode_X_attr[NUM_ATTR_REGS * 2] = {
    0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03,
    0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07,
    0x08, 0x08, 0x09, 0x09, 0x0A, 0x0A, 0x0B, 0x0B,
    0x0C, 0x0C, 0x0D, 0x0D, 0x0E, 0x0E, 0x0F, 0x0F,
    0x10, 0x41, 0x11, 0x00, 0x12, 0x0F, 0x13, 0x00,
    0x14, 0x00, 0x15, 0x00
};
static unsigned short mode_X_graphics[NUM_GRAPHICS_REGS] = {
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x4005, 0x0506, 0x0F07,
    0xFF08	
};
/* VGA register settings for text mode 3(color text) */
static unsigned short text_seq[NUM_SEQUENCER_REGS] = {
    0x0100, 0x2001, 0x0302, 0x0003, 0x0204
};
static unsigned short text_CRTC[NUM_CRTC_REGS] = {
    0x5F00, 0x4F01, 0x5002, 0x8203, 0x5504, 0x8105, 0xBF06, 0x0F07,
    0x0008, 0x0F09, 0x0D0A, 0x0E0B, 0x000C, 0x000D, 0x000E, 0x000F,
    0x9C10, 0x8E11, 0x8F12, 0x2813, 0x1F14, 0x9615, 0xB916, 0xA317,
    0x0018
};
static unsigned char text_attr[NUM_ATTR_REGS * 2] = {
    0x00, 0x00, 0x01, 0x01, 0x02, 0x02, 0x03, 0x03,
    0x04, 0x04, 0x05, 0x05, 0x06, 0x06, 0x07, 0x07,
    0x08, 0x08, 0x09, 0x09, 0x0A, 0x0A, 0x0B, 0x0B,
    0x0C, 0x0C, 0x0D, 0x0D, 0x0E, 0x0E, 0x0F, 0x0F,
    0x10, 0x03, 0x11, 0x00, 0x12, 0x01, 0x13, 0x08,
    0x14, 0x00, 0x15, 0x00
};
static unsigned short text_graphics[NUM_GRAPHICS_REGS] = {
    0x0000, 0x0001, 0x0002, 0x0003, 0x0004, 0x1005, 0x0E06, 0x0007,
    0xFF08
};

static void set_seq_regs_and_reset(unsigned short table[NUM_SEQUENCER_REGS], unsigned char val) {
    /*
     * Dump table of values to sequencer registers.    Includes forced reset
     * as well as video blanking.
     */
    REP_OUTSW(0x03C4, table, NUM_SEQUENCER_REGS);

    /* Delay a bit... */
    { volatile int ii; for (ii = 0; ii < 10000; ii++); }

    /* Set VGA miscellaneous output register. */
    outb(0x03C2, val);

    /* Turn sequencer on(array values above should always force reset). */
    OUTW(0x03C4, 0x0300);
}

static void set_CRTC_registers(unsigned short table[NUM_CRTC_REGS]) {
    /* clear protection bit to enable write access to first few registers */
    OUTW(0x03D4, 0x0011);
    REP_OUTSW(0x03D4, table, NUM_CRTC_REGS);
}

static void set_attr_registers(unsigned char table[NUM_ATTR_REGS * 2]) {
    /* Reset attribute register to write index next rather than data. */
    asm volatile("          \n\
        inb (%%dx), %%al    \n\
        "
        :
        : "d"(0x03DA)
        : "eax", "memory"
    );
    REP_OUTSB(0x03C0, table, NUM_ATTR_REGS * 2);
}

static void set_graphics_registers(unsigned short table[NUM_GRAPHICS_REGS]) {
    REP_OUTSW(0x03CE, table, NUM_GRAPHICS_REGS);
}

static void write_font_data() {
    int i;                /* loop index over characters                   */
    int j;                /* loop index over font bytes within characters */
    unsigned char* fonts; /* pointer into video memory                    */

    /* Prepare VGA to write font data into video memory. */
    OUTW(0x3C4, 0x0402);
    OUTW(0x3C4, 0x0704);
    OUTW(0x3CE, 0x0005);
    OUTW(0x3CE, 0x0406);
    OUTW(0x3CE, 0x0204);

    /* Copy font data from array into video memory. */
    for (i = 0, fonts = mem_image; i < 256; i++) {
        for (j = 0; j < 16; j++) {
            fonts[j] = font_data[i][j];
        }
        fonts += 32; /* skip 16 bytes between characters */
    }

    /* Prepare VGA for text mode. */
    OUTW(0x3C4, 0x0302);
    OUTW(0x3C4, 0x0304);
    OUTW(0x3CE, 0x1005);
    OUTW(0x3CE, 0x0E06);
    OUTW(0x3CE, 0x0004);
}

static void VGA_blank(int blank_bit) {
    /*
     * Move blanking bit into position for VGA sequencer register
     *(index 1).
     */
    blank_bit = ((blank_bit & 1) << 5);

    asm volatile("                                                      \n\
        movb $0x01, %%al        /* Set sequencer index to 1 */          \n\
        movw $0x03C4, %%dx                                              \n\
        outb %%al, (%%dx)                                               \n\
        incw %%dx                                                       \n\
        inb (%%dx), %%al        /* Read old value */                    \n\
        andb $0xDF, %%al        /* Calculate new value */               \n\
        orl %0,%%eax                                                    \n\
        outb %%al, (%%dx)       /* Write new value */                   \n\
        movw $0x03DA, %%dx      /* Enable display(0x20->P[0x3C0]) */    \n\
        inb (%%dx), %%al        /* Set attr reg state to index */       \n\
        movw $0x03C0, %%dx      /* Write index 0x20 to enable */        \n\
        movb $0x20, %%al                                                \n\
        outb %%al, (%%dx)                                               \n\
        "
        :
        : "g"(blank_bit)
        : "eax", "edx", "memory"
    );
}
static void fill_palette_text() {
    /* 6-bit RGB(red, green, blue) values VGA colors and grey scale */
    static unsigned char palette_RGB[32][3] = {
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F}, 
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F}, 
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F}, 
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
    };

    /* Start writing at color 0. */
    outb(0x03C8, 0x00);

    /* Write all 32 colors from array. */
    REP_OUTSB(0x03C9, palette_RGB, 32 * 3);
}
static void fill_palette_mode_x() {
    /* 6-bit RGB(red, green, blue) values for first 64 colors */
    /* these are coded for 2 bits red, 2 bits green, 2 bits blue */
   /* static unsigned char palette_RGB[64][3] = {
        {0x00, 0x00, 0x00}, {0x00, 0x00, 0x15},
        {0x00, 0x00, 0x2A}, {0x00, 0x00, 0x3F},
        {0x00, 0x15, 0x00}, {0x00, 0x15, 0x15},
        {0x00, 0x15, 0x2A}, {0x00, 0x15, 0x3F},
        {0x00, 0x2A, 0x00}, {0x00, 0x2A, 0x15},
        {0x00, 0x2A, 0x2A}, {0x00, 0x2A, 0x3F},
        {0x00, 0x3F, 0x00}, {0x00, 0x3F, 0x15},
        {0x00, 0x3F, 0x2A}, {0x00, 0x3F, 0x3F},
        {0x15, 0x00, 0x00}, {0x15, 0x00, 0x15},
        {0x15, 0x00, 0x2A}, {0x15, 0x00, 0x3F},
        {0x15, 0x15, 0x00}, {0x15, 0x15, 0x15},
        {0x15, 0x15, 0x2A}, {0x15, 0x15, 0x3F},
        {0x15, 0x2A, 0x00}, {0x15, 0x2A, 0x15},
        {0x15, 0x2A, 0x2A}, {0x15, 0x2A, 0x3F},
        {0x15, 0x3F, 0x00}, {0x15, 0x3F, 0x15},
        {0x15, 0x3F, 0x2A}, {0x15, 0x3F, 0x3F},
        {0x2A, 0x00, 0x00}, {0x2A, 0x00, 0x15},
        {0x2A, 0x00, 0x2A}, {0x2A, 0x00, 0x3F},
        {0x2A, 0x15, 0x00}, {0x2A, 0x15, 0x15},
        {0x2A, 0x15, 0x2A}, {0x2A, 0x15, 0x3F},
        {0x2A, 0x2A, 0x00}, {0x2A, 0x2A, 0x15},
        {0x2A, 0x2A, 0x2A}, {0x2A, 0x2A, 0x3F},
        {0x2A, 0x3F, 0x00}, {0x2A, 0x3F, 0x15},
        {0x2A, 0x3F, 0x2A}, {0x2A, 0x3F, 0x3F},
        {0x3F, 0x00, 0x00}, {0x3F, 0x00, 0x15},
        {0x3F, 0x00, 0x2A}, {0x3F, 0x00, 0x3F},
        {0x3F, 0x15, 0x00}, {0x3F, 0x15, 0x15},
        {0x3F, 0x15, 0x2A}, {0x3F, 0x15, 0x3F},
        {0x3F, 0x2A, 0x00}, {0x3F, 0x2A, 0x15},
        {0x3F, 0x2A, 0x2A}, {0x3F, 0x2A, 0x3F},
        {0x3F, 0x3F, 0x00}, {0x3F, 0x3F, 0x15},
        {0x3F, 0x3F, 0x2A}, {0x3F, 0x3F, 0x3F}
    };*/
	
	  static unsigned char palette_RGB[64][3] = {
		{0x0, 0x0, 0x0},
        {0, 0x30, 0}, {0x3F, 0x20, 0},//first place for backgrond, following four reserved for font and cursur
        {0, 0x30, 0x30}, {0, 0x20, 0},
        {60, 61, 35}, {55, 9, 4},
        {59, 62, 35}, {0, 35, 9},
        {63, 63, 40}, {44, 54, 27},
        {16, 42, 16}, {3, 11, 32},
        {17, 35, 23}, {32, 36, 35},
        {63, 58, 30}, {36, 43, 37},
        {61, 9, 11}, {25, 24, 29},
        {33, 50, 24}, {2, 11, 11},
        {20, 25, 31}, {50, 52, 52},
        {13, 11, 15}, {16, 11, 10},
        {27, 9, 12}, {31, 8, 8},
        {28, 13, 8}, {46, 56, 51},
        {42, 12, 10}, {26, 20, 25},
        {1, 39, 12}, {58, 9, 5},
        {49, 33, 29}, {29, 38, 38},
        {12, 39, 20}, {31, 39, 32},
        {58, 28, 27}, {14, 13, 12},
        {55, 38, 41}, {40, 18, 15},
        {29, 12, 13}, {62, 6, 4},
        {1, 58, 16}, {9, 48, 25},
        {47, 54, 49}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0x3F, 0x3F, 0x3F}, {0x3F, 0x3F, 0x3F},
        {0, 0, 0x3F}
    };


    /* Start writing at color 0. */
    outb(0x03C8, 0x00);

    /* Write all 64 colors from array. */
    REP_OUTSB(0x03C9, palette_RGB, 64 * 3);
}

int set_mode_X() {
    /*
     * The code below was produced by recording a call to set mode 0013h
     * with display memory clearing and a windowed frame buffer, then
     * modifying the code to set mode X instead. The code was then
     * generalized into functions...
     *
     * modifications from mode 13h to mode X include...
     * Sequencer Memory Mode Register: 0x0E to 0x06(0x3C4/0x04)
     * Underline Location Register   : 0x40 to 0x00(0x3D4/0x14)
     * CRTC Mode Control Register    : 0xA3 to 0xE3(0x3D4/0x17)
     */

    VGA_blank(1);                               /* blank the screen      */
    set_seq_regs_and_reset(mode_X_seq, 0x63);   /* sequencer registers   */
    set_CRTC_registers(mode_X_CRTC);            /* CRT control registers */
    set_attr_registers(mode_X_attr);            /* attribute registers   */
    set_graphics_registers(mode_X_graphics);    /* graphics registers    */
	fill_palette_mode_x();
	//SET_WRITE_MASK(0x0F00);
	memset(mem_image, 0, MODE_X_MEM_SIZE);
    VGA_blank(0);                               /* unblank the screen    */
	//SET_WRITE_MASK(0x0F00);
	target_img ^= 0x4000;
	OUTW(0x03D4, (target_img & 0xFF00) | 0x0C);
    OUTW(0x03D4, ((target_img & 0x00FF) << 8) | 0x0D);

    /* Return success. */
    return 0;
}
void set_text_mode_3(int clear_scr) {
	unsigned long* txt_scr; /* pointer to text screens in video memory */
    int i;
    VGA_blank(1);           /* blank the screen */

    /*
     * The value here had been changed to 0x63, but seems to work
     * fine in QEMU(and VirtualPC, where I got it) with the 0x04
     * bit set(VGA_MIS_DCLK_28322_720).
     */
    set_seq_regs_and_reset(text_seq, 0x63);  /* sequencer registers     */
    set_CRTC_registers(text_CRTC);           /* CRT control registers   */
    set_attr_registers(text_attr);           /* attribute registers     */
    set_graphics_registers(text_graphics);   /* graphics registers      */
    fill_palette_text();                     /* palette colors          */
   if (clear_scr) {                         /* clear screens if needed */
        txt_scr = (unsigned long*)(mem_image + 0x18000);
        for (i = 0; i <15440 ; i++) {//8192
            *txt_scr++ = 0x07200720;
        }
    }
    write_font_data();   /* copy fonts to video mem */
    VGA_blank(0);        /* unblank the screen      */
}

/* tab:4
 *
 * photo.c - photo display functions
 *
 * "Copyright (c) 2011 by Steven S. Lumetta."
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without written agreement is
 * hereby granted, provided that the above copyright notice and the following
 * two paragraphs appear in all copies of this software.
 *
 * IN NO EVENT SHALL THE AUTHOR OR THE UNIVERSITY OF ILLINOIS BE LIABLE TO
 * ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES ARISING OUT  OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF THE AUTHOR AND/OR THE UNIVERSITY OF ILLINOIS HAS BEEN ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * THE AUTHOR AND THE UNIVERSITY OF ILLINOIS SPECIFICALLY DISCLAIM ANY
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE
 * PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND NEITHER THE AUTHOR NOR
 * THE UNIVERSITY OF ILLINOIS HAS ANY OBLIGATION TO PROVIDE MAINTENANCE,
 * SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS."
 *
 * Author:        Steve Lumetta
 * Version:       3
 * Creation Date: Fri Sep 9 21:44:10 2011
 * Filename:      photo.c
 * History:
 *    SL    1    Fri Sep 9 21:44:10 2011
 *        First written(based on mazegame code).
 *    SL    2    Sun Sep 11 14:57:59 2011
 *        Completed initial implementation of functions.
 *    SL    3    Wed Sep 14 21:49:44 2011
 *        Cleaned up code for distribution.
 */


#include <string.h>

#include "assert.h"
#include "modex.h"
#include "photo.h"
#include "photo_headers.h"
#include "world.h"


#define ZERO		0			/*constant of 0*/
#define ONE			1			/*constant of 1*/
#define TWO			2			/*constant of 2*/
#define THREE		3			/*constant of 3*/
#define FOUR		4			/*constant of 4*/
#define FIVE		5			/*constant of 5*/
#define SIX			6			/*constant of 6*/
#define SEVEN		7			/*constant of 7*/
#define EIGHT		8			/*constant of 8*/
#define NINE		9			/*constant of 9*/
#define ELEVEN		11			/*constant of 11*/
#define TWELVE		12			/*constant of 12*/
#define FOURTEEN	14			/*constant of 14*/
#define SIXTEEN		16			/*constant of 16*/
#define SEVENTEEN		17			/*constant of 17*/
#define EIGHTEEN		18			/*constant of 18*/
#define NINETEEN		19			/*constant of 19*/
#define TWENTYFOUR		24			/*constant of 24*/
#define TWENTYFIVE		25			/*constant of 25*/
#define TWENTYSIX		26			/*constant of 26*/
#define TWENTYSEVEN		27			/*constant of 27*/
#define SIXTYFOUR		64			/*constant of 64*/
#define ONETWOEIGHT		128			/*constant of 128*/
#define TWOFIVESIX		256			/*constant of 256*/
#define ARRAYSIZE		4096			/*constant of 4096*/
#define SINGLEBITMASK			0x1			/*constant of 0x1*/
#define FOURBITMASK				0xF			/*constant of 0xF*/
#define ZEROTWOBITMASK			0x00			/*constant of 0x00*/
#define LASTTWOBITMASK			0x3		/*constant of 0x3*/
#define ONEFBITMASK				0x1F		/*constant of 0x1F*/
#define THREEFBITMASK			0x3F		/*constant of 0x3F*/

/* types local to this file(declared in types.h) */

/*
 * A room photo.  Note that you must write the code that selects the
 * optimized palette colors and fills in the pixel data using them as
 * well as the code that sets up the VGA to make use of these colors.
 * Pixel data are stored as one-byte values starting from the upper
 * left and traversing the top row before returning to the left of
 * the second row, and so forth.  No padding should be used.
 */
struct photo_t {
    photo_header_t hdr;            /* defines height and width */
    uint8_t        palette[192][3];     /* optimized palette colors */
    uint8_t*       img;                 /* pixel data               */
};

/*
 * An object image.  The code for managing these images has been given
 * to you.  The data are simply loaded from a file, where they have
 * been stored as 2:2:2-bit RGB values(one byte each), including
 * transparent pixels(value OBJ_CLR_TRANSP).  As with the room photos,
 * pixel data are stored as one-byte values starting from the upper
 * left and traversing the top row before returning to the left of the
 * second row, and so forth.  No padding is used.
 */
struct image_t {
    photo_header_t hdr;  /* defines height and width */
    uint8_t*       img;  /* pixel data               */
};

/* this is the struct we used to build in the array. It includes index, counter and red, green and blue value 
 * which all have 2 versions. One is 5 bits or 6 bits, the other one is 6 bits.*/
typedef struct sortnode {
	int Rvalue;
	int Gvalue;
	int Bvalue;
	int RRvalue;
	int GGvalue;
	int BBvalue;
	int counter;
	int index;
}sortnode;

/* file-scope variables */

/*
 * The room currently shown on the screen.  This value is not known to
 * the mode X code, but is needed when filling buffers in callbacks from
 * that code(fill_horiz_buffer/fill_vert_buffer).  The value is set
 * by calling prep_room.
 */
static const room_t* cur_room = NULL;


/*
 * fill_horiz_buffer
 *   DESCRIPTION: Given the(x,y) map pixel coordinate of the leftmost
 *                pixel of a line to be drawn on the screen, this routine
 *                produces an image of the line.  Each pixel on the line
 *                is represented as a single byte in the image.
 *
 *                Note that this routine draws both the room photo and
 *                the objects in the room.
 *
 *   INPUTS:(x,y) -- leftmost pixel of line to be drawn
 *   OUTPUTS: buf -- buffer holding image data for the line
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void fill_horiz_buffer(int x, int y, unsigned char buf[SCROLL_X_DIM]) {
    int            idx;   /* loop index over pixels in the line          */
    object_t*      obj;   /* loop index over objects in the current room */
    int            imgx;  /* loop index over pixels in object image      */
    int            yoff;  /* y offset into object image                  */
    uint8_t        pixel; /* pixel from object image                     */
    const photo_t* view;  /* room photo                                  */
    int32_t        obj_x; /* object x position                           */
    int32_t        obj_y; /* object y position                           */
    const image_t* img;   /* object image                                */

    /* Get pointer to current photo of current room. */
    view = room_photo(cur_room);

    /* Loop over pixels in line. */
    for (idx = 0; idx < SCROLL_X_DIM; idx++) {
        buf[idx] = (0 <= x + idx && view->hdr.width > x + idx ? view->img[view->hdr.width * y + x + idx] : 0);
    }

    /* Loop over objects in the current room. */
    for (obj = room_contents_iterate(cur_room); NULL != obj; obj = obj_next(obj)) {
        obj_x = obj_get_x(obj);
        obj_y = obj_get_y(obj);
        img = obj_image(obj);

        /* Is object outside of the line we're drawing? */
        if (y < obj_y || y >= obj_y + img->hdr.height || x + SCROLL_X_DIM <= obj_x || x >= obj_x + img->hdr.width) {
            continue;
        }

        /* The y offset of drawing is fixed. */
        yoff = (y - obj_y) * img->hdr.width;

        /*
         * The x offsets depend on whether the object starts to the left
         * or to the right of the starting point for the line being drawn.
         */
        if (x <= obj_x) {
            idx = obj_x - x;
            imgx = 0;
        }
        else {
            idx = 0;
            imgx = x - obj_x;
        }

        /* Copy the object's pixel data. */
        for (; SCROLL_X_DIM > idx && img->hdr.width > imgx; idx++, imgx++) {
            pixel = img->img[yoff + imgx];

            /* Don't copy transparent pixels. */
            if (OBJ_CLR_TRANSP != pixel) {
                buf[idx] = pixel;
            }
        }
    }
}
/*
 * cmpfunc
 *   DESCRIPTION: helper compare function for qsort.
 *   INPUTS:const void * a: a pointer to the first node we want to sort
 *			const void * b: a pointer to the second node we want to sort
 *   OUTPUTS: none
 *   RETURN VALUE: difference of value in two nodes.
 *   SIDE EFFECTS: none
 */
int cmpfunc (const void * a, const void * b) {
	int c = ((struct sortnode *)a)->counter;
	int d = ((struct sortnode *)b)->counter;
   return ( d - c );
}


/*
 * fill_vert_buffer
 *   DESCRIPTION: Given the(x,y) map pixel coordinate of the top pixel of
 *                a vertical line to be drawn on the screen, this routine
 *                produces an image of the line.  Each pixel on the line
 *                is represented as a single byte in the image.
 *
 *                Note that this routine draws both the room photo and
 *                the objects in the room.
 *
 *   INPUTS:(x,y) -- top pixel of line to be drawn
 *   OUTPUTS: buf -- buffer holding image data for the line
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void fill_vert_buffer(int x, int y, unsigned char buf[SCROLL_Y_DIM]) {
    int            idx;   /* loop index over pixels in the line          */
    object_t*      obj;   /* loop index over objects in the current room */
    int            imgy;  /* loop index over pixels in object image      */
    int            xoff;  /* x offset into object image                  */
    uint8_t        pixel; /* pixel from object image                     */
    const photo_t* view;  /* room photo                                  */
    int32_t        obj_x; /* object x position                           */
    int32_t        obj_y; /* object y position                           */
    const image_t* img;   /* object image                                */

    /* Get pointer to current photo of current room. */
    view = room_photo(cur_room);

    /* Loop over pixels in line. */
    for (idx = 0; idx < SCROLL_Y_DIM; idx++) {
        buf[idx] = (0 <= y + idx && view->hdr.height > y + idx ? view->img[view->hdr.width *(y + idx) + x] : 0);
    }

    /* Loop over objects in the current room. */
    for (obj = room_contents_iterate(cur_room); NULL != obj; obj = obj_next(obj)) {
        obj_x = obj_get_x(obj);
        obj_y = obj_get_y(obj);
        img = obj_image(obj);

        /* Is object outside of the line we're drawing? */
        if (x < obj_x || x >= obj_x + img->hdr.width ||
            y + SCROLL_Y_DIM <= obj_y || y >= obj_y + img->hdr.height) {
            continue;
        }

        /* The x offset of drawing is fixed. */
        xoff = x - obj_x;

        /*
         * The y offsets depend on whether the object starts below or
         * above the starting point for the line being drawn.
         */
        if (y <= obj_y) {
            idx = obj_y - y;
            imgy = 0;
        }
        else {
            idx = 0;
            imgy = y - obj_y;
        }

        /* Copy the object's pixel data. */
        for (; SCROLL_Y_DIM > idx && img->hdr.height > imgy; idx++, imgy++) {
            pixel = img->img[xoff + img->hdr.width * imgy];

            /* Don't copy transparent pixels. */
            if (OBJ_CLR_TRANSP != pixel) {
                buf[idx] = pixel;
            }
        }
    }
}


/*
 * image_height
 *   DESCRIPTION: Get height of object image in pixels.
 *   INPUTS: im -- object image pointer
 *   OUTPUTS: none
 *   RETURN VALUE: height of object image im in pixels
 *   SIDE EFFECTS: none
 */
uint32_t image_height(const image_t* im) {
    return im->hdr.height;
}


/*
 * image_width
 *   DESCRIPTION: Get width of object image in pixels.
 *   INPUTS: im -- object image pointer
 *   OUTPUTS: none
 *   RETURN VALUE: width of object image im in pixels
 *   SIDE EFFECTS: none
 */
uint32_t image_width(const image_t* im) {
    return im->hdr.width;
}

/*
 * photo_height
 *   DESCRIPTION: Get height of room photo in pixels.
 *   INPUTS: p -- room photo pointer
 *   OUTPUTS: none
 *   RETURN VALUE: height of room photo p in pixels
 *   SIDE EFFECTS: none
 */
uint32_t photo_height(const photo_t* p) {
    return p->hdr.height;
}


/*
 * photo_width
 *   DESCRIPTION: Get width of room photo in pixels.
 *   INPUTS: p -- room photo pointer
 *   OUTPUTS: none
 *   RETURN VALUE: width of room photo p in pixels
 *   SIDE EFFECTS: none
 */
uint32_t photo_width(const photo_t* p) {
    return p->hdr.width;
}


/*
 * prep_room
 *   DESCRIPTION: Prepare a new room for display.  You might want to set
 *                up the VGA palette registers according to the color
 *                palette that you chose for this room.
 *   INPUTS: r -- pointer to the new room
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: changes recorded cur_room for this file
 */
void prep_room(const room_t* r) {
	cur_room = r;
	add_palette_color(room_photo(r)->palette);
}


/*
 * read_obj_image
 *   DESCRIPTION: Read size and pixel data in 2:2:2 RGB format from a
 *                photo file and create an image structure from it.
 *   INPUTS: fname -- file name for input
 *   OUTPUTS: none
 *   RETURN VALUE: pointer to newly allocated photo on success, or NULL
 *                 on failure
 *   SIDE EFFECTS: dynamically allocates memory for the image
 */
image_t* read_obj_image(const char* fname) {
    FILE*    in;        /* input file               */
    image_t* img = NULL;    /* image structure          */
    uint16_t x;            /* index over image columns */
    uint16_t y;            /* index over image rows    */
    uint8_t  pixel;        /* one pixel from the file  */

    /*
     * Open the file, allocate the structure, read the header, do some
     * sanity checks on it, and allocate space to hold the image pixels.
     * If anything fails, clean up as necessary and return NULL.
     */
    if (NULL == (in = fopen(fname, "r+b")) ||
        NULL == (img = malloc(sizeof (*img))) ||
        NULL != (img->img = NULL) || /* false clause for initialization */
        1 != fread(&img->hdr, sizeof (img->hdr), 1, in) ||
        MAX_OBJECT_WIDTH < img->hdr.width ||
        MAX_OBJECT_HEIGHT < img->hdr.height ||
        NULL == (img->img = malloc
        (img->hdr.width * img->hdr.height * sizeof (img->img[0])))) {
        if (NULL != img) {
            if (NULL != img->img) {
                free(img->img);
            }
            free(img);
        }
        if (NULL != in) {
            (void)fclose(in);
        }
        return NULL;
    }

    /*
     * Loop over rows from bottom to top.  Note that the file is stored
     * in this order, whereas in memory we store the data in the reverse
     * order(top to bottom).
     */
    for (y = img->hdr.height; y-- > 0; ) {

        /* Loop over columns from left to right. */
        for (x = 0; img->hdr.width > x; x++) {

            /*
             * Try to read one 8-bit pixel.  On failure, clean up and
             * return NULL.
             */
            if (1 != fread(&pixel, sizeof (pixel), 1, in)) {
                free(img->img);
                free(img);
                (void)fclose(in);
                return NULL;
            }

            /* Store the pixel in the image data. */
            img->img[img->hdr.width * y + x] = pixel;
        }
    }

    /* All done.  Return success. */
    (void)fclose(in);
    return img;
}


/*
 * read_photo
 *   DESCRIPTION: Read size and pixel data in 5:6:5 RGB format from a
 *                photo file and create a photo structure from it.
 *                This function first collect the frequency of using each octree node.
 *				  then sort them in descending order and take first 128 nodes' value to palette.
 *				  Then the function loops around the rest nodes and take average values and store
 *				  them to the 64 layer two nodes. These are the last 64 palette values.
 *				  Finally, this function loops around the pixel again to set them a new palette index value.
 *   INPUTS: fname -- file name for input
 *   OUTPUTS: none
 *   RETURN VALUE: pointer to newly allocated photo on success, or NULL
 *                 on failure
 *   SIDE EFFECTS: dynamically allocates memory for the photo
 */
photo_t* read_photo(const char* fname) {
    FILE*    in;    /* input file               */
    photo_t* p = NULL;    /* photo structure          */
    uint16_t x;        /* index over image columns */
    uint16_t y;        /* index over image rows    */
    uint16_t pixel;    /* one pixel from the file  */

    /*
     * Open the file, allocate the structure, read the header, do some
     * sanity checks on it, and allocate space to hold the photo pixels.
     * If anything fails, clean up as necessary and return NULL.
     */
    if (NULL == (in = fopen(fname, "r+b")) ||
        NULL == (p = malloc(sizeof (*p))) ||
        NULL != (p->img = NULL) || /* false clause for initialization */
        1 != fread(&p->hdr, sizeof (p->hdr), 1, in) ||
        MAX_PHOTO_WIDTH < p->hdr.width ||
        MAX_PHOTO_HEIGHT < p->hdr.height ||
        NULL == (p->img = malloc
        (p->hdr.width * p->hdr.height * sizeof (p->img[0])))) {
        if (NULL != p) {
            if (NULL != p->img) {
                free(p->img);
            }
            free(p);
        }
        if (NULL != in) {
            (void)fclose(in);
        }
        return NULL;
    }

    /*
     * Loop over rows from bottom to top.  Note that the file is stored
     * in this order, whereas in memory we store the data in the reverse
     * order(top to bottom).
     */
	 int red, green, black;
	 int count = ZERO;
	 int value;
	 
	 /* initialize the array of struct we declared above.*/
	 struct sortnode photosort[ARRAYSIZE] = {{ZERO}, {ZERO}, {ZERO}, {ZERO}, {ZERO}, {ZERO}, {ZERO}, {ZERO}};
    for (y = p->hdr.height; y-- > 0; ) {

        /* Loop over columns from left to right. */
        for (x = 0; p->hdr.width > x; x++) {

            /*
             * Try to read one 16-bit pixel.  On failure, clean up and
             * return NULL.
             */
            if (1 != fread(&pixel, sizeof (pixel), 1, in)) {
                free(p->img);
                free(p);
                (void)fclose(in);
                return NULL;
            }
			
			/* load the four most significant bits of R,G and B to local variables.*/
			red = (pixel >> TWELVE) & FOURBITMASK;
			green = (pixel >> SEVEN) & FOURBITMASK;
			black = (pixel >> ONE) & FOURBITMASK;
			
			/* locate the value to one of the layer four node of octree.*/
			int indexv = red * TWOFIVESIX + green * SIXTEEN + black;
			
			/* store all values to this node struct and counter is incremented.*/
			photosort[indexv].Rvalue += (pixel >> TWELVE) & FOURBITMASK;
			photosort[indexv].Gvalue += (pixel >> SEVEN) & FOURBITMASK;
			photosort[indexv].Bvalue += (pixel >> ONE) & FOURBITMASK;
			photosort[indexv].counter++;
			
			/* the five bit R and B, 6 bit G value is first been averaged before stored into the node.*/
			photosort[indexv].RRvalue = (photosort[indexv].RRvalue * (photosort[indexv].counter - ONE) + ((pixel >> ELEVEN) & ONEFBITMASK)) / photosort[indexv].counter;
			photosort[indexv].GGvalue = (photosort[indexv].GGvalue * (photosort[indexv].counter - ONE) + ((pixel >> FIVE) & THREEFBITMASK)) / photosort[indexv].counter;
			photosort[indexv].BBvalue = (photosort[indexv].BBvalue * (photosort[indexv].counter - ONE) + ((pixel >> ZERO) & ONEFBITMASK)) / photosort[indexv].counter;		
        }
    }
	
	/* storing the index of each node in this new while loop.*/
	count = ZERO;
	while (count < ARRAYSIZE)
	{
		photosort[count].index = count;
		count++;
	}
	
	/* call the qsort function to change photosort to a array with descending order in counter.*/
	qsort(photosort, ARRAYSIZE, sizeof (sortnode), cmpfunc);
	count = ZERO;
	int column = ZERO;
	
	/* store the first 128 nodes' 4 bit value to the first 128 rows of the palette.*/
	while (count < ONETWOEIGHT)
	{
		p->palette[count][column] = (unsigned char)((photosort[count].Rvalue / photosort[count].counter) << TWO);
		p->palette[count][column + ONE] = (unsigned char)((photosort[count].Gvalue / photosort[count].counter) << TWO);
		p->palette[count][column + TWO] = (unsigned char)((photosort[count].Bvalue / photosort[count].counter) << TWO);
		count++;
	}
	red = ZERO;
	green = ZERO;
	black = ZERO;
	count = ONETWOEIGHT;
	
	/* initialize new array of 64 to store the 5bits and 6 bits color value which stored in layer two of octree.*/
	int layerR[SIXTYFOUR] = {ZERO};
	int layerG[SIXTYFOUR] = {ZERO};
	int layerB[SIXTYFOUR] = {ZERO};
	int layerc[SIXTYFOUR] = {ZERO};
	
	/* go through all the rest nodes to store the value and counters to these four arrays.*/
	while (count < ARRAYSIZE)
	{
		red = (photosort[count].RRvalue >> THREE) & LASTTWOBITMASK;
		green = (photosort[count].GGvalue >> FOUR) & LASTTWOBITMASK;
		black = (photosort[count].BBvalue >> THREE) & LASTTWOBITMASK;
		
		/* calculate the index that which node in layer two matches the current node index.*/
		value = red * SIXTEEN + green * FOUR + black;
		layerR[value] += photosort[count].RRvalue;
		layerG[value] += photosort[count].GGvalue;
		layerB[value] += photosort[count].BBvalue;
		layerc[value] += ONE;
		count++;
	}
	count = ZERO;
	
	/* create a new loop to go through the 4 arrays we created above, and set their average valus to the last 64 palette colors.*/
	while (count < SIXTYFOUR)
	{
		/* store their value to palette if counter is not 0.*/
		if (layerc[count] != ZERO)
		{
			p->palette[count + ONETWOEIGHT][column] = (unsigned char)((layerR[count] / layerc[count]) << ONE);
			p->palette[count + ONETWOEIGHT][column + ONE] = (unsigned char)(layerG[count] / layerc[count]);
			p->palette[count + ONETWOEIGHT][column + TWO] = (unsigned char)((layerB[count] / layerc[count]) << ONE);
			count++;
			continue;
		}
		/* store 0 to palette if counter is 0.*/
		else if (layerc[count] == ZERO)
		{
			p->palette[count + ONETWOEIGHT][column] = (unsigned char)(ZEROTWOBITMASK);
			p->palette[count + ONETWOEIGHT][column + ONE] = (unsigned char)(ZEROTWOBITMASK);
			p->palette[count + ONETWOEIGHT][column + TWO] = (unsigned char)(ZEROTWOBITMASK);
			count++;
		}
	}
	count = ZERO;
	
	/* reset the pixel pointer to the first pixel of the picture.*/
	fseek(in, ZERO, SEEK_SET);
	
	/* skip the first pixel of the room.*/
	fread(&p->hdr, sizeof (p->hdr), ONE, in);
	int i;
	int flag = ZERO;
	
	/* go through the picture to reset their palette index.*/
	for (y = p->hdr.height; y-- > ZERO; ) {

        for (x = ZERO; p->hdr.width > x; x++) {
			
			/*
             * Try to read one 16-bit pixel.  On failure, clean up and
             * return NULL.
             */
			if (ONE != fread(&pixel, sizeof (pixel), ONE, in)) {
                free(p->img);
                free(p);
                (void)fclose(in);
                return NULL;
            }
				/* read their original palette index.*/
				red = (pixel >> TWELVE) & FOURBITMASK;
				green = (pixel >> SEVEN) & FOURBITMASK;
				black = (pixel >> ONE) & FOURBITMASK;
				
				/* achieve the original palette index.*/
				int indexv = red * TWOFIVESIX + green * SIXTEEN + black;
				
				/* first check if the newly assigned index would be amond the first 128 palette indexes.*/
				for (i = ZERO; i < ONETWOEIGHT; i++)
				{
					if (photosort[i].index == indexv)
						{
							/* set new index to p->img if found.*/
							p->img[p->hdr.width * y + x] = i + SIXTYFOUR;
							
							/* set flag to 1 to prevent inefficiently searching in the following 64 palettes.*/
							flag = ONE; 
							break;
						}
					
				}
				
				/* if first 128 indexes are not matched, then searching the last 64 palettes.*/
				if (flag == ZERO)
				{
					red = (pixel >> FOURTEEN) & LASTTWOBITMASK;
					green = (pixel >> NINE) & LASTTWOBITMASK;
					black = (pixel >> THREE) & LASTTWOBITMASK;
					
					/* calculate the new palette index that should be assigned.*/
					indexv = red * SIXTEEN + green * FOUR + black;
					
					/* set the new indexes to p->image.*/
					p->img[p->hdr.width * y + x] = indexv + ONETWOEIGHT + SIXTYFOUR;
				}
				
				/* reset the flag.*/
				flag = ZERO;
		}
	 }
    /* All done.  Return success. */
    (void)fclose(in);
    return p;
}

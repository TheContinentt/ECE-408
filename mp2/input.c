/* tab:4
 *
 * input.c - source file for input control to maze game
 *
 * "Copyright (c) 2004-2011 by Steven S. Lumetta."
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
 * Version:       7
 * Creation Date: Thu Sep 9 22:25:48 2004
 * Filename:      input.c
 * History:
 *    SL    1    Thu Sep 9 22:25:48 2004
 *        First written.
 *    SL    2    Sat Sep 12 14:34:19 2009
 *        Integrated original release back into main code base.
 *    SL    3    Sun Sep 13 03:51:23 2009
 *        Replaced parallel port with Tux controller code for demo.
 *    SL    4    Sun Sep 13 12:49:02 2009
 *        Changed init_input order slightly to avoid leaving keyboard
 *        in odd state on failure.
 *    SL    5    Sun Sep 13 16:30:32 2009
 *        Added a reasonably robust direct Tux control for demo mode.
 *    SL    6    Wed Sep 14 02:06:41 2011
 *        Updated input control and test driver for adventure game.
 *    SL    7    Wed Sep 14 17:07:38 2011
 *        Added keyboard input support when using Tux kernel mode.
 */

#include <ctype.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/io.h>
#include <termio.h>
#include <termios.h>
#include <unistd.h>

#include "module/tuxctl-ioctl.h"
#include "assert.h"
#include "input.h"

#define ZERO		0			/*constant of 0*/
#define SIXTY		60			/*constant of 60*/
#define FIVENINE	59			/*constant of 59*/
#define TEN			10			/*constant of 10*/
#define EIGHT		8			/*constant of 8*/
#define TWELVE		12			/*constant of 12*/
#define FOUR		4			/*constant of 4*/
#define ONEBITMASK			0x7F			/*constant of 0x7F*/
#define TWOBITMASK			0xBF			/*constant of 0xBF*/
#define THREEBITMASK			0xDF			/*constant of 0xDF*/
#define FOURBITMASK			0xEF			/*constant of 0xEF*/
#define FIVEBITMASK			0xF7			/*constant of 0xF7*/
#define SIXBITMASK			0xFB			/*constant of 0xFB*/
#define SEVENBITMASK			0xFD			/*constant of 0xFD*/
#define EIGHTBITMASK			0xFE			/*constant of 0xFE*/
#define BUTTONVALUEONE			0x04070000			/*constant of 0x04070000*/
#define BUTTONVALUETWO			0x040F0000			/*constant of 0x040F0000*/
#define BUTTONVALUETHREE			0x101189AB			/*constant of 0x101189AB*/

/* set to 1 and compile this file by itself to test functionality */
#define TEST_INPUT_DRIVER 0

/* set to 1 to use tux controller; otherwise, uses keyboard input */
#define USE_TUX_CONTROLLER 1

/* stores original terminal settings */
static struct termios tio_orig;
static int fd;
/*
 * init_input
 *   DESCRIPTION: Initializes the input controller.  As both keyboard and
 *                Tux controller control modes use the keyboard for the quit
 *                command, this function puts stdin into character mode
 *                rather than the usual terminal mode.
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: 0 on success, -1 on failure
 *   SIDE EFFECTS: changes terminal settings on stdin; prints an error
 *                 message on failure
 */
int init_input() {
    struct termios tio_new;
	/*int fd = open("/dev/ttyS0", O_RDWR | O_NOCTTY);
	int ldisc_num = N_MOUSE;
	ioctl(fd, TIOCSETD, &ldisc_num);
	ioctl(fd, TUX_INIT);*/
    /*
     * Set non-blocking mode so that stdin can be read without blocking
     * when no new keystrokes are available.
     */
    if (fcntl(fileno(stdin), F_SETFL, O_NONBLOCK) != 0) {
        perror("fcntl to make stdin non-blocking");
        return -1;
    }

    /*
     * Save current terminal attributes for stdin.
     */
    if (tcgetattr(fileno(stdin), &tio_orig) != 0) {
        perror("tcgetattr to read stdin terminal settings");
        return -1;
    }

    /*
     * Turn off canonical(line-buffered) mode and echoing of keystrokes
     * to the monitor.  Set minimal character and timing parameters so as
     * to prevent delays in delivery of keystrokes to the program.
     */
    tio_new = tio_orig;
    tio_new.c_lflag &= ~(ICANON | ECHO);
    tio_new.c_cc[VMIN] = 1;
    tio_new.c_cc[VTIME] = 0;
    if (tcsetattr(fileno(stdin), TCSANOW, &tio_new) != 0) {
        perror("tcsetattr to set stdin terminal settings");
        return -1;
    }

    /* Return success. */
    return 0;
}

static char typing[MAX_TYPED_LEN + 1] = {'\0'};

const char* get_typed_command() {
    return typing;
}

void reset_typed_command() {
    typing[0] = '\0';
}

static int32_t valid_typing(char c) {
    /* Valid typing include letters, numbers, space, and backspace/delete. */
    return (isalpha(c) || isdigit(c) || ' ' == c || 8 == c || 127 == c);
}

static void typed_a_char(char c) {
    int32_t len = strlen(typing);

    if (8 == c || 127 == c) {
        if (0 < len) {
            typing[len - 1] = '\0';
        }
    }
    else if (MAX_TYPED_LEN > len) {
        typing[len] = c;
        typing[len + 1] = '\0';
    }
}

/*
 * get_command
 *   DESCRIPTION: Reads a command from the input controller.  As some
 *                controllers provide only absolute input(e.g., go
 *                right), the current direction is needed as an input
 *                to this routine.
 *   INPUTS: cur_dir -- current direction of motion
 *   OUTPUTS: none
 *   RETURN VALUE: command issued by the input controller
 *   SIDE EFFECTS: drains any keyboard input
 */
cmd_t get_command() {

/* enable keyboard and tux at the same time.*/
#if (USE_TUX_CONTROLLER == 1) /* use keyboard control with arrow keys */
    static int state = 0;                 /* small FSM for arrow keys */
#endif
    static cmd_t command = CMD_NONE;
    cmd_t pushed = CMD_NONE;
    int ch;

    /* Read all characters from stdin. */
    while ((ch = getc(stdin)) != EOF) {

        /* Backquote is used to quit the game. */
        if (ch == '`')
            return CMD_QUIT;

/* enable keyboard and tux at the same time.*/
#if (USE_TUX_CONTROLLER == 1) /* use keyboard control with arrow keys */

        /*
         * Arrow keys deliver the byte sequence 27, 91, and 'A' to 'D';
         * we use a small finite state machine to identify them.
         *
         * Insert, home, and page up keys deliver 27, 91, '2'/'1'/'5' and
         * then a tilde.  We recognize the digits and don't check for the
         * tilde.
         */
        switch (state) {
            case 0:
                if (27 == ch) {
                    state = 1;
                }
                else if (valid_typing(ch)) {
                    typed_a_char(ch);
                }
                else if (10 == ch || 13 == ch) {
                    pushed = CMD_TYPED;
                }
                break;
            case 1:
                if (91 == ch) {
                    state = 2;
                }
                else {
                    state = 0;
                    if (valid_typing(ch)) {
                        /*
                         * Note that we may be discarding an ESC(27), but
                         * we don't use that as typed input anyway.
                         */
                        typed_a_char(ch);
                    }
                    else if (10 == ch || 13 == ch) {
                        pushed = CMD_TYPED;
                    }
                }
                break;
            case 2:
                if (ch >= 'A' && ch <= 'D') {
                    switch (ch) {
                        case 'A': pushed = CMD_UP;    break;
                        case 'B': pushed = CMD_DOWN;  break;
                        case 'C': pushed = CMD_RIGHT; break;
                        case 'D': pushed = CMD_LEFT;  break;
                    }
                    state = 0;
                }
                else if (ch == '1' || ch == '2' || ch == '5') {
                    switch (ch) {
                        case '2': pushed = CMD_MOVE_LEFT;  break;
                        case '1': pushed = CMD_ENTER;      break;
                        case '5': pushed = CMD_MOVE_RIGHT; break;
                    }
                    state = 3; /* Consume a '~'. */
                }
                else {
                    state = 0;
                    if (valid_typing(ch)) {
                        /*
                         * Note that we may be discarding an ESC(27) and
                         * a bracket(91), but we don't use either as
                         * typed input anyway.
                         */
                        typed_a_char(ch);
                    }
                    else if (10 == ch || 13 == ch) {
                        pushed = CMD_TYPED;
                    }
                }
                break;
            case 3:
                state = 0;
                if ('~' == ch) {
                    /* Consume it silently. */
                }
                else if (valid_typing(ch)) {
                    typed_a_char(ch);
                }
                else if (10 == ch || 13 == ch) {
                    pushed = CMD_TYPED;
                }
                break;
        }
#else /* USE_TUX_CONTROLLER */
        /* Tux controller mode; still need to support typed commands. */
        if (valid_typing(ch)) {
            typed_a_char(ch);
        }
        else if (10 == ch || 13 == ch) {
            pushed = CMD_TYPED;
        }
#endif /* USE_TUX_CONTROLLER */
    }

    /*
     * Once a direction is pushed, that command remains active
     * until a turn is taken.
     */
    if (pushed == CMD_NONE) {
        command = CMD_NONE;
    }
    return pushed;
}

/*
 * get_button_tux
 *   DESCRIPTION: calls the ioctl function and then return the 8-bit button value to adventure.c
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: 8-bit button command number
 *   SIDE EFFECTS: none
 */
uint8_t get_button_tux()
{
	uint8_t ch;
	
	/*call ioctl function in tuxctl-ioctl.c.*/
	ioctl(fd, TUX_BUTTONS, &ch);
	return ch;
}

/*
 * get_command_tux
 *   DESCRIPTION: Reads a command from the tux controller. By calling the 
 *				  ioctl functions in tuxctl-ioctl.c, we would get the button 8-bit
 *				  and then return the correct command.
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: command issued by the tux controller
 *   SIDE EFFECTS: drains any tux input
 */
cmd_t get_command_tux()
{
	
	/* initialize return command to CMD_NONE*/
    cmd_t pushed = CMD_NONE;
	uint8_t ch;
	
	/* get the return 8-bit value by call ioctl function.*/
	ioctl(fd, TUX_BUTTONS, &ch);
	
	/* since button is active-low, we could use switch and case to
	know what the tux command is.*/
    switch (ch) {
	/* return value is 0x01111111*/
	case ONEBITMASK:
		pushed = CMD_RIGHT;
		break;
	/* return value is 0x10111111*/
	case TWOBITMASK:
		pushed = CMD_LEFT;
		break;
	/* return value is 0x11011111*/
	case THREEBITMASK:
		pushed = CMD_DOWN;
		break;
	/* return value is 0x11101111*/
	case FOURBITMASK:
		pushed = CMD_UP;
		break;
	/* return value is 0x11110111*/
	case FIVEBITMASK:
		pushed = CMD_MOVE_RIGHT;
		break;
	/* return value is 0x11111011*/
	case SIXBITMASK:
		pushed = CMD_ENTER;
		break;
	/* return value is 0x11111101*/
	case SEVENBITMASK:
		pushed = CMD_MOVE_LEFT;
		break;
	/* return value is 0x11111110*/
	case EIGHTBITMASK:
		pushed = CMD_QUIT;
		break;
	}
    return pushed;
}

/*
 * shutdown_input
 *   DESCRIPTION: Cleans up state associated with input control.  Restores
 *                original terminal settings.
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: restores original terminal settings
 */
void shutdown_input() {
    (void)tcsetattr(fileno(stdin), TCSANOW, &tio_orig);
}


/*
 * display_time_on_tux
 *   DESCRIPTION: Show number of elapsed seconds as minutes:seconds
 *                on the Tux controller's 7-segment displays.
 *   INPUTS: num_seconds -- total seconds elapsed so far
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: changes state of controller's display
 */
void display_time_on_tux(int num_seconds) {
#if (USE_TUX_CONTROLLER != 0)
//#error "Tux controller code is not operational yet."

	/* initialize variables.*/
	int quotient;
	int hourreminder;
	int reminder;
	int callvalue;
	int lastdigit;
	quotient = num_seconds / SIXTY;
	
	/* achieve value of led 2.*/
	hourreminder = quotient % TEN;
	
	/*achieve the values of led0 and led1.*/
	reminder = num_seconds - quotient * SIXTY;
	
	/*achieve the value of led3.*/
	quotient = (quotient - hourreminder) / TEN;
	callvalue = BUTTONVALUEONE;
	
	/*achieve the value of led0.*/
	lastdigit = reminder % TEN;
	
	/*achieve the value of led1.*/
	reminder = (reminder - lastdigit) / TEN;
	
	/* shift each value to their corresponding bit location.*/
	reminder = reminder << FOUR;
	
	/* if led3's value is 0, we set it to be down.*/
	if (quotient != ZERO)
	{
		quotient  = quotient << TWELVE;
		callvalue = BUTTONVALUETWO;
	}
	hourreminder = hourreminder << EIGHT;
	
	/* add them together.*/
	callvalue += quotient;
	callvalue += reminder;
	callvalue += lastdigit;
	callvalue += hourreminder;
	
	/*call the ioctl set_led function.*/
	ioctl(fd, TUX_SET_LED, callvalue);
	return;
#endif
}

/*
 * initializetux
 *   DESCRIPTION: open the serial port and set the Tux controller line discipline.
 *				  then call the ioctl initialization function.
 *   INPUTS: none
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: clear the button stored value and led stored value.
 */
void initializetux()
{
	fd = open("/dev/ttyS0", O_RDWR | O_NOCTTY);
	int ldsic_num = N_MOUSE;
	ioctl(fd, TIOCSETD, &ldsic_num);
	
	/* call initialization ioctl function.*/
	ioctl(fd, TUX_INIT);
	return;
}

/*
 * testled
 *   DESCRIPTION: this function is created to test led spam. In this infinite loop, 
 *				  I call set_led function each loop. And the driver is not crashed.
 *   INPUTS: the first test number that I would like to print
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: none
 */
void testled(int quotient)
{
	while (1)
	{
		ioctl(fd, TUX_SET_LED, quotient);
		quotient++;
	}
}

#if (TEST_INPUT_DRIVER == 1)
int main() {
    cmd_t last_cmd = CMD_NONE;
    cmd_t cmd;
    static const char* const cmd_name[NUM_COMMANDS] = {
        "none", "right", "left", "up", "down", "move left",
        "enter", "move right", "typed command", "quit"
    };

    /* Grant ourselves permission to use ports 0-1023 */
    if (ioperm(0, 1024, 1) == -1) {
        perror("ioperm");
        return 3;
    }
	
	/* first open the port and initialize the driver before operation on it.*/
	initializetux();
	display_time_on_tux(FIVENINE);
	/* set led value by call ioctl function.*/
	ioctl(fd, TUX_SET_LED, BUTTONVALUETHREE);
    init_input();
    while (1) {
        while ((cmd = get_command_tux()) == last_cmd);
        last_cmd = cmd;
        printf("command issued: %s\n", cmd_name[cmd]);
        if (cmd == CMD_QUIT)
            break;
    }
    shutdown_input();
    return 0;
}

#endif

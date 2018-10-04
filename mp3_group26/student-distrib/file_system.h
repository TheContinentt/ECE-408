
#ifndef _FILE_SYSTEM_H
#define _FILE_SYSTEM_H
#include "types.h"
#include "lib.h"
#include "terminal.h"

#define FILE_NAME_L    		32
#define RESERVE_DEN_L      	24
#define RESERVE_BOO_L  		52
#define DIR_E_L				63
#define DATA_B_IDX			1023



/* dentry_t is a struct we build to store all the data of dentry in a file system.
 * So we could access data just by changing our index of each variables instead of 
 * calculate complex offset.*/
typedef struct dentry_t {
		uint8_t file_name[FILE_NAME_L];
		int32_t file_type;
		int32_t inode_idx;
		uint8_t reserved[RESERVE_DEN_L];
}dentry_t;

/* boot_block_t is a struct we build to store all the data of boot_block in a file system.
 * So we could access each dentry just by changing our index of each variables instead of 
 * calculate complex offset.*/
typedef struct boot_block_t {
		int32_t dir_num;
		int32_t inode_num;
		int32_t data_block_num;
		int8_t 	reserved[RESERVE_BOO_L];
		dentry_t dir_entry[DIR_E_L];
}boot_block_t;

/* inode_block_t is a struct we build to store all the data of inode block in a file system.
 * So we could access inode and file data stored in them just by changing our index of each 
 * variables instead of calculate complex offset.*/
typedef struct inode_block_t {
		int32_t length;
		int32_t data_block_idx[DATA_B_IDX];
}inode_block_t;

/* Initialize the file system.*/
void file_sys_init(int32_t start_addr);

/* read the dentry data by a given filename pointer.*/
int32_t read_dentry_by_name (const uint8_t* fname, dentry_t* dentry);

/* read the dentry data by a given dentry index.*/
int32_t read_dentry_by_index(uint32_t index, dentry_t* dentry);

/* read the file data by given data offset, length, buffer amd an inode index.*/
int32_t read_data(uint32_t inode, uint32_t offset, uint8_t* buf, uint32_t length);

/* get the size of the file.*/
int32_t get_file_size(int32_t index);

/* externt the jumptable for directory.*/
file_operation_table_t DIR_jumptable;

/* externt the jumptable for regular file.*/
file_operation_table_t FILE_jumptable;

/* write data to the directory.*/
int32_t dir_write(int32_t fd, const void * buf, int32_t nbytes);

/* read the data from the specific directory.*/
int32_t dir_read(int32_t fd, void* buf, int32_t nbytes);

/* close the directory that opened previously.*/
int32_t dir_close(int32_t fd);

/* open a directory by a giben filename pointer.*/
int32_t dir_open(const uint8_t* filename);

/* write data to the file.*/
int32_t file_write(int32_t fd, const void * buf, int32_t nbytes);

/* read the data from the specific file.*/
int32_t file_read(int32_t fd, void* buf, int32_t nbytes);

/* close the file that opened previously.*/
int32_t file_close(int32_t fd);

/* open a file by a giben filename pointer.*/
int32_t file_open(const uint8_t* filename);


#endif

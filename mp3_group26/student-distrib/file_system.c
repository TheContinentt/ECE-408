
#include "file_system.h"

/* initialize global variables to store that data of files and directory, thus they could be used in  multiple function calls.*/
boot_block_t* boot_block;
inode_block_t* inode_addr;
dentry_t curr_dir_dentry;
dentry_t curr_file_dentry;

int last_round;
#define FOURBYTES		32
#define FOURKILOBYTES		4096
#define ONE				1
#define EIGHT			8
/*
 * file_sys_init
 *   DESCRIPTION: initialzies the file system by setting the starting address to the boot_block and inode struct we defined.
 *   INPUTS: start_addr: the starting address of the whole file system.
 *   OUTPUTS: none
 *   RETURN VALUE: none
 *   SIDE EFFECTS: set start_addr and 4kb after it as starting address of two structs.
 */
void file_sys_init(int32_t start_addr){

	/* set starting address to pointer boot_block, 4kb after starting address to inode_addr.*/
	boot_block = (boot_block_t*)start_addr;
	inode_addr = (inode_block_t*)(start_addr+FOURKILOBYTES);
	
	/*fill in the jump table*/
	DIR_jumptable.read = dir_read;
	DIR_jumptable.write = dir_write;
	DIR_jumptable.open = dir_open;
	DIR_jumptable.close = dir_close;
	/*fill in the jump table*/
	FILE_jumptable.read = file_read;
	FILE_jumptable.write = file_write;
	FILE_jumptable.open = file_open;
	FILE_jumptable.close = file_close;
}

/*
 * read_dentry_by_name
 *   DESCRIPTION: find and read the info of a dentry by matching its filename with input filename.
 *				  Once successfully founded, store info to the input dentry struct and return 0.
 *				  Otherwise, return -1.
 *   INPUTS: fname: a pointer to the filename which we want to find in the file system.
 *				    dentry: a dentry struct where we want to store the dentry we found.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: if read dentry by name and copy it successfully.
 *				   -1: if no file found, return -1.
 *   SIDE EFFECTS: copy dentry info in file structure to the dentry structure we defined.
 */
int32_t read_dentry_by_name (const uint8_t* fname, dentry_t* dentry) {

	/* initialize variables anf flags.*/
	int i;
	int j;
	int cmp_flag = 1;
	int found = 0;
	int found_index;
	int len;
	int curr_file_len;

	/* check if file name is a null pointer, return -1 if it is.*/
	if (fname == NULL) return -1;

	/* go through the loop to check which diresctory entry fits the file name we wwant to find.*/
	for (i=0; i<boot_block->dir_num; i++) {
		len =strlen((int8_t*)fname);
		curr_file_len = strlen((int8_t*)(boot_block->dir_entry[i]).file_name);
		if (len>32) len = 32;
		if (curr_file_len >32) curr_file_len = 32;
		/* first we compare current length of current dir entry's file name to that of we want to find.
		* if they didn't match, the loop continues to the next loop.*/
		if (len != curr_file_len)
			continue;

		/*if(len>FOURBYTES)
			len = FOURBYTES;*/
		/* else we check the characters of file name one by one, once one character doesn't match,
		* we set flag value to 0 and break.*/
		for (j=0; j<len; j++) {
			if (fname[j] != *((boot_block->dir_entry[i]).file_name+j)) {
				cmp_flag = 0;
				break;
			}
		}

		/* if flag shows that we find the file, set found flag to 1, and break the outer searching loop.*/
		if (cmp_flag) {
			found = 1;

			/* save the file index to found_index.*/
			found_index = i;
			break;
		}

		/* if not match current dir entry's file name, reset cmp_flag and go to next loop.*/
		else cmp_flag = 1;
	}

	/* if we find the directory entry that contains the file we want, we need to copy this dentry's information
	* to the input dentry struct. So we could store them and read them.*/
	if (found) {

		/* use loop to copy the file name's character one by one.*/
		for (i=0; i<FOURBYTES; i++) {
			dentry->file_name[i] = (boot_block->dir_entry[found_index]).file_name[i];
		}

		/* copy file_type and inode_idx.*/
		dentry->file_type = (boot_block->dir_entry[found_index]).file_type;
		dentry->inode_idx = (boot_block->dir_entry[found_index]).inode_idx;

		/* return 0 since find successfully.*/
		return 0;
	}

	/* if file not found, return -1.*/
	return -1;
}

/*
 * read_dentry_by_index
 *   DESCRIPTION: find and read the info of a dentry by matching its dentry index with input filename.
 *				  Once successfully founded, store info to the input dentry struct and return 0.
 *				  Otherwise, return -1.
 *   INPUTS: index: the index of dentry which we want to find.
 *			 dentry: a dentry struct where we want to store the dentry we found.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: if read dentry by index and copy it successfully.
 *				   -1: if read fails, return -1.
 *   SIDE EFFECTS: copy dentry info in file structure to the dentry structure we defined.
 */
int32_t read_dentry_by_index(uint32_t index, dentry_t* dentry) {

	int i;

	/* if index is out of range, return -1.*/
	if (index < 0 || index >= boot_block->dir_num) return -1;

	/* copy filename in the directory entry to the input dentry struct.*/
	for (i=0; i<FOURBYTES; i++) {
			dentry->file_name[i] = (boot_block->dir_entry[index]).file_name[i];
		}

	/* copy file_type and inode_idx and return 0.*/
	dentry->file_type = (boot_block->dir_entry[index]).file_type;
	dentry->inode_idx = (boot_block->dir_entry[index]).inode_idx;
	return 0;
}

/*
 * read_data
 *   DESCRIPTION: find and read the data of a specific inode by given inode index.
 *				  Once successfully readed, store data to the input buffer.
 *   INPUTS: inode: the index of inode of the file system.
 *			 offset: offset in the file we read. We read file data start from this offset.
 *			 buf: uint8_t buffer that stores the data we read.
 *			 length: the length we want to read in this file
 *   OUTPUTS: none
 *   RETURN VALUE: 0: if read till the end of the inode block, we return 0.
 *				   -1: if read fails, return -1.
 *				   length: if we don't reach the end of file and complete read, return length.
 *   SIDE EFFECTS: copy file data to the parameter buffer.
 */
int32_t read_data(uint32_t inode,  uint32_t offset, uint8_t* buf, uint32_t length) {

	/* if inode index is out of range or the file in this inode is empty, return -1.*/
	if (inode < 0 || inode >= boot_block->inode_num) return -1;
	if (inode_addr[inode].length == 0) return -1;

	/* initialize variables and pointer to the inode where we read data.*/
	int i=0;
	int ret;
	uint8_t* data_block_head = (uint8_t*)(inode_addr + boot_block->inode_num);

	/* calculate the starting offset in the data block and which datablock to start from.*/
	uint32_t offset_data_block_idx = offset/FOURKILOBYTES;
	uint32_t offset_data_addr = offset % FOURKILOBYTES;

	/* initialize variables used in the following copy loop.*/
	int32_t curr_data_idx;
	int32_t curr_data_addr = offset_data_addr;
	int32_t copy_length;

	/* chech if offset plus length is less than the file's length, set copy_length to the
	 * parameter length if it is.*/
	if (offset+length <= inode_addr[inode].length) {
		copy_length = length;
		ret = length;
	}

	/* if offset plus length is larger than the file's length, set copy length to the length file length minus offset.*/
	else {
		copy_length = (inode_addr[inode].length)-offset;

		/* set return value to 0 to show we reaches the end of the file.*/
		ret = 0;
	}

	/* use loop to copy the file data to the input buffer.*/
	while (copy_length != 0) {

		/* set start index of data block.*/
		curr_data_idx = inode_addr[inode].data_block_idx[offset_data_block_idx];

		/* check if the index is out of range, return -1 if it is.*/
		if (curr_data_idx < 0 || curr_data_idx >= boot_block->data_block_num)  return -1;

		/* copy data in data block to input buffer one byte by one byte.*/
		while (curr_data_addr < FOURKILOBYTES) {

			/* load current one byte data to buffer.*/
			buf[i]=*(data_block_head+curr_data_idx*FOURKILOBYTES+curr_data_addr);

			/* increase index in buffer and index in data block we in to go to the next loop. Also decrease the length we need to copy.*/
			i++;
			curr_data_addr++;
			copy_length--;

			/* break if no more data need to be copied.*/
			if (copy_length == 0) break;
		}
			/* Once one data block is read to the end, reset index and start reading the next data block in this inode.*/
			curr_data_addr = 0;
			offset_data_block_idx++;
	}

	/* return return value.*/
	return ret;
}

/* get file size helper function*/
int32_t get_file_size(int32_t index) {
	return ((inode_block_t*)(inode_addr+index))->length;
}

/*
 * file_open
 *   DESCRIPTION: open the file with given file name by calling read_dentry_by_name.
 *   INPUTS: filename: the file name of file that we want to open.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this open function.
 *   SIDE EFFECTS: copy dentry info to the global dentry_t struct curr_file_dentry for following read_data.
 */
int32_t file_open(const uint8_t* filename){
	last_round = 0;
	return read_dentry_by_name(filename, &curr_file_dentry);
}

/*
 * file_close
 *   DESCRIPTION: close the file.
 *   INPUTS: fd: an int32_t integer.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this close function.
 *   SIDE EFFECTS: none.
 */
int32_t file_close(int32_t fd){
	return 0;
}

/*
 * file_read
 *   DESCRIPTION: read the file and data and load data to the given buffer parameter.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer where we store our read data.
 *			 nbytes: the length of data that we want to read from the file.
 *   OUTPUTS: output the read file data to terminal by calling terminal_write function.
 *   RETURN VALUE: 0: return 0 to where calls this open function.
 *   SIDE EFFECTS: output the read file data to terminal by calling terminal_write function.
 */
int32_t file_read(int32_t fd, void* buf, int32_t nbytes){

	/* load buf to a uint8_t pointer buf_temp.*/
	
	int len;
	/*int i;
	int curr_read;
	
	for (i=0; i<EIGHT; i++) {
		if (phys_addr_available[i] == ONE) break;
	}
	curr_read = i-ONE;*/
	
	/* load the current PCB address.*/
	//PCB_t* curr_read_PCB = (PCB_t*) (EIGHTMB-(curr_read+ONE)*(EIGHTKB));
	
	/* get the read length of data in the given file where we stored in the global variable.*/
	//len = read_data(curr_file_dentry.inode_idx,0,buf_temp,nbytes);
	len = read_data(((file_descriptor_t*)fd)->inode, ((file_descriptor_t*)fd)->file_position , (uint8_t*)buf, nbytes);

	/* if return value from read_data is 0, reset data length to the length of data in this inode.*/
	/*if (len == 0) {
		len = (inode_addr[(file_descriptor_t*)fd->inode]).length;
	}*/
	((file_descriptor_t*)fd)->file_position = ((file_descriptor_t*)fd)->file_position+len;
	if ((len == 0) && (last_round == 0)) {
		last_round = 1;
		return inode_addr[((file_descriptor_t*)fd)->inode].length-((file_descriptor_t*)fd)->file_position;
	} 
	else if ((len == 0) && (last_round == 1)) {
		last_round = 0;
		return 0;
	}
	
	/* write the data to terminal.*/
	return len;
}

/*
 * file_write
 *   DESCRIPTION: write data to the file.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer where data stored in it need to be written to file.
 *			 nbytes: the length of data that we want to write to the file.
 *   OUTPUTS: none
 *   RETURN VALUE: -1: return -1 to where calls this close function.
 *   SIDE EFFECTS: none.
 */
int32_t file_write(int32_t fd, const void * buf, int32_t nbytes){
	return -1;
}

/*
 * dir_open
 *   DESCRIPTION: open the directory  y its given name and store its data to global variable curr_dir_dentry.
 *   INPUTS: filename: the file name of directory that we want to open.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 if open successfully.
 *				   -1: it the file name given is not a name of a directory.
 *   SIDE EFFECTS: copy directory info to the global dentry_t struct curr_file_dentry for following read_data.
 */
int32_t dir_open(const uint8_t* filename){
	read_dentry_by_name(filename,&curr_dir_dentry);

	/* chech if the read file is a directory or not.*/
	if(curr_dir_dentry.file_type!=1)
		return -1;
	return 0;
}

/*
 * dir_close
 *   DESCRIPTION: close the directory.
 *   INPUTS: fd: an int32_t integer.
 *   OUTPUTS: none
 *   RETURN VALUE: 0: return 0 to where calls this close function.
 *   SIDE EFFECTS: none.
 */
int32_t dir_close(int32_t fd){
	return 0;
}

/* initialize this global variable to 0.*/
int directory_index = 0;

/*
 * dir_read
 *   DESCRIPTION: read the directory and data and load data to the given buffer parameter.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer where we store our read data.
 *			 nbytes: the length of data that we want to read from the directory.
 *   OUTPUTS: none.
 *   RETURN VALUE: j: the length of filename that we've written to the parameter buffer.
 *				   -1: if read directory fails.
 *   SIDE EFFECTS: none.
 */
int32_t dir_read(int32_t fd, void* buf, int32_t nbytes){
	int j = 0;
	int ret;

	/* check if read directory by index workds fine.*/
	ret = read_dentry_by_index(directory_index,&curr_dir_dentry);
	directory_index++;

	/* return -1 if read fails.*/
	if(ret==-1){
		directory_index = 0;
	return 0;}

	/* read the string length of the directory's filename.*/
	ret = strlen((int8_t*)(curr_dir_dentry.file_name));

	/* use loop to load the name to the parameter buffer. Break if there's at least 32 chars are stored.*/
	for(j = 0;j<ret;j++){
		if(j == FOURBYTES)
			break;
		*(uint8_t*)(buf+j) = curr_dir_dentry.file_name[j];
	}

	/* return the number of characters that we have written to the buffer.*/
	return j;
}

/*
 * dir_write
 *   DESCRIPTION: write data to the directory.
 *   INPUTS: fd: an int32_t integer.
 *			 buf: a buffer where data stored in it need to be written to directory.
 *			 nbytes: the length of data that we want to write to the directory.
 *   OUTPUTS: none
 *   RETURN VALUE: -1: return -1 to where calls this close function.
 *   SIDE EFFECTS: none.
 */
int32_t dir_write(int32_t fd, const void * buf, int32_t nbytes){
	return -1;
}

CC=gcc
CFLAGS=-I. -O3 -std=c99

all: test bruteforce

test: test.c tiger_impl.c
	$(CC) $(CFLAGS) -o test test.c tiger_impl.c

bruteforce: tiger_bruteforce.c tiger_impl.c
	$(CC) $(CFLAGS) -o bruteforce tiger_bruteforce.c tiger_impl.c

clean:
	rm -f test bruteforce
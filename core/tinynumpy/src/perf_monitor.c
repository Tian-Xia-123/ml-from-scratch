#define _GNU_SOURCE
#include "perf_monitor.h"
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>

int setup_perf_event(int type, int config) {
  struct perf_event_attr pe = {.type = type,
                               .size = sizeof(struct perf_event_attr),
                               .config = config,
                               .disabled = 1,
                               .exclude_kernel = 1,
                               .exclude_hv = 1};

  int fd = syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
  if (fd < 0) {
    perror("Error: perf_event_open failed.");
  }

  return fd;
}

void start_perf_event(int fd) {
  if (fd < 0) {
    fprintf(stderr, "Error: Cannot stop. Invalid file descriptor (fd=%d).\n",
            fd);
  }
  ioctl(fd, PERF_EVENT_IOC_RESET, 0);
  ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

void stop_perf_event(int fd) {
  if (fd < 0) {
    fprintf(stderr, "Error: Cannot stop. Invalid file descriptor (fd=%d).\n",
            fd);
  }
  ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
}

long long read_perf_event(int fd) {
  if (fd < 0) {
    fprintf(stderr, "Error: Cannot read. Invalid file descriptor (fd=%d).\n",
            fd);
  }

  long long count;
  if (read(fd, &count, sizeof(long long)) != sizeof(long long)) {
    perror("Error: Failed to read perf event value.");
    return -1;
  }
  return count;
}
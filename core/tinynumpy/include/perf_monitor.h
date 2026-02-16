#ifndef PERF_MONITOR_H
#define PERF_MONITOR_H

// Initializes the perf event
int setup_perf_event(int type, int config);

// Starts the perf event
void start_perf_event(int fd);

// Stops the perf event
void stop_perf_event(int fd);

// Reads the value of the perf event
long long read_perf_event(int fd);

#endif
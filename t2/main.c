#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

 
void producer_consumer_parallel(int size, int *vec, int n, int n_threads) {
	int i, j;
	int *buffer;
	long long unsigned int sum = 0;

	/*
	 * Use pragma omp parallel for with n_threads
	 * Shared variables size, vec, n, i
	 * Privated variables j, buffer
	 * Reduction used to protect crital section in sum += buffer[j]
	*/
	#pragma omp parallel for num_threads(n_threads) default(none) shared(size, vec, n, i) private(j, buffer) reduction(+:sum)
	for(i=0;i<n;i++) {
		// Per iteration of for, create new buffer
		buffer = (int *)malloc(size*sizeof(int));
		if(i % 2 == 0) {	// PRODUTOR
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMER
			for(j=0;j<size;j++) {
				sum += buffer[j];
			}
		}
		// Free memory used by buffer
		free(buffer);
	}
	printf("%llu\n",sum);
}




int main(int argc, char const *argv[])
{
	int n_threads, size, n, *vec, i;
	double start, end;

	//Scan NThreads
	scanf("%d", &n_threads);

	//Scan number of iterations
	scanf("%d", &n);

	//Scan size of buffer
	scanf("%d", &size);

	//Create vec
	vec = (int *)malloc(n*sizeof(int));

	//Scan vec
	for (i = 0; i < n; ++i)
	{
		scanf("%d", &vec[i]);
	}

	//Get start time
	start = omp_get_wtime();
	// Call function producer consumer
	producer_consumer_parallel(size, vec, n, n_threads);
	// Get end time
	end = omp_get_wtime();

	// print duration of function
	printf("%lf\n", end - start);

	//Free memory used by vec
	free(vec);

	return 0;
}

/*
cat /proc/cpuinfo 

processor	: 0
vendor_id	: GenuineIntel
cpu family	: 6
model		: 58
model name	: Intel(R) Core(TM) i5-3230M CPU @ 2.60GHz
stepping	: 9
microcode	: 0x17
cpu MHz		: 1200.000
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 2
apicid		: 0
initial apicid	: 0
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
bogomips	: 5187.89
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 1
vendor_id	: GenuineIntel
cpu family	: 6
model		: 58
model name	: Intel(R) Core(TM) i5-3230M CPU @ 2.60GHz
stepping	: 9
microcode	: 0x17
cpu MHz		: 1200.000
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 1
cpu cores	: 2
apicid		: 2
initial apicid	: 2
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
bogomips	: 5187.89
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 2
vendor_id	: GenuineIntel
cpu family	: 6
model		: 58
model name	: Intel(R) Core(TM) i5-3230M CPU @ 2.60GHz
stepping	: 9
microcode	: 0x17
cpu MHz		: 1600.000
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 0
cpu cores	: 2
apicid		: 1
initial apicid	: 1
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
bogomips	: 5187.89
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:

processor	: 3
vendor_id	: GenuineIntel
cpu family	: 6
model		: 58
model name	: Intel(R) Core(TM) i5-3230M CPU @ 2.60GHz
stepping	: 9
microcode	: 0x17
cpu MHz		: 1200.000
cache size	: 3072 KB
physical id	: 0
siblings	: 4
core id		: 1
cpu cores	: 2
apicid		: 3
initial apicid	: 3
fpu		: yes
fpu_exception	: yes
cpuid level	: 13
wp		: yes
flags		: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc aperfmperf eagerfpu pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm ida arat epb xsaveopt pln pts dtherm tpr_shadow vnmi flexpriority ept vpid fsgsbase smep erms
bogomips	: 5187.89
clflush size	: 64
cache_alignment	: 64
address sizes	: 36 bits physical, 48 bits virtual
power management:
*/

/*
lscpu

Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                4
On-line CPU(s) list:   0-3
Thread(s) per core:    2
Core(s) per socket:    2
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 58
Stepping:              9
CPU MHz:               1200.000
BogoMIPS:              5187.89
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              3072K
NUMA node0 CPU(s):     0-3
*/

/*
gprof

Each sample counts as 0.01 seconds.
  %   cumulative   self              self     total           
 time   seconds   seconds    calls   s/call   s/call  name    
100.64      3.55     3.55        1     3.55     3.55  main
  0.00      3.55     0.00        1     0.00     0.00  producer_consumer_parallel

*/

/*
arq1.in
-O0
0.812508195 = 0.006197/0.007627

-O1 
3.137721519 = 0.006197/0.001975

-O2
4.109416446 = 0.006197/0.001508

-O3
5.431200701 = 0.006197/0.001141


ar2.in
-O0
1.023941455 = 0.292152/0.285321

-O1 
3.784499398 = 0.292152/0.077197

-O2
4.560884226 = 0.292152/0.064056

-O3
5.912093249 = 0.292152/0.049416


ar3.in
-O0
0.994978921 = 2.825767/2.840027

-O1 
3.953846993 = 2.825767/0.714688

-O2
5.155107462 = 2.825767/0.548149

-O3
6.942387681 = 2.825767/0.407031

*/
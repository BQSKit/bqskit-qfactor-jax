OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[7], q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[9];
cx q[7], q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[9];
cx q[7], q[9];
cx q[5], q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[5];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
cx q[5], q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[5];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
cx q[5], q[7];
cx q[4], q[5];
cx q[7], q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[5];
u3(0.0, 0.0, 0.7853981633974483) q[7];
cx q[4], q[5];
cx q[6], q[7];
u3(0.0, 0.0, 5.497787143782138) q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[5];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[6];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
cx q[4], q[5];
cx q[6], q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[6];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
cx q[2], q[4];
cx q[6], q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[2];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
u3(1.5707963267948966, 0.0, 6.283185307179586) q[6];
cx q[7], q[8];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[3], q[4];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[8];
cx q[0], q[2];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[3];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
cx q[7], q[8];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[0];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[2];
cx q[3], q[4];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[8];
cx q[0], q[2];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[3];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
cx q[7], q[8];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[0];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[2];
cx q[3], q[4];
cx q[7], q[9];
cx q[0], q[2];
cx q[1], q[3];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[4];
u3(1.5707963267948966, 0.0, -3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[1];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[3];
cx q[6], q[7];
u3(0.0, 0.0, 7.0685834705770345) q[9];
cx q[1], q[3];
u3(1.5707963267948966, 2.356194490192345, -3.141592653589793) q[6];
u3(1.5707963267948966, -2.356194490192345, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[1];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[3];
cx q[7], q[9];
cx q[1], q[3];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[1];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[3];
cx q[7], q[9];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[9];
cx q[7], q[9];
cx q[6], q[7];
u3(0.0, 0.0, 5.497787143782138) q[7];
u3(1.5707963267948966, 0.0, 3.141592653589793) q[7];
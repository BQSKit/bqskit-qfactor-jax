OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[1], q[8];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 6.25864161457342) q[0];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.02454369260617026) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 6.23409792196725) q[0];
u3(0.0, 0.0, 0.0245436999999999) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[0], q[7];
cx q[1], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[0], q[3];
cx q[1], q[8];
u3(0.0, 0.0, 5.497787143782138) q[3];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[3];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 6.25864161457342) q[8];
u3(0.0, 0.0, 0.02454369260617026) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[0], q[8];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 6.25864161457342) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[0];
u3(0.0, 0.0, 6.25864165358979) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[7];
cx q[1], q[0];
u3(0.0, 0.0, 6.23409792196725) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[0], q[3];
cx q[1], q[8];
u3(0.0, 0.0, 0.7853981633974483) q[3];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[3];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.02454369260617026) q[8];
u3(0.0, 0.0, 6.25864161457342) q[0];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[0], q[8];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.02454369260617026) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 6.25864161457342) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 6.23409792196725) q[0];
u3(0.0, 0.0, 0.02454369260617026) q[8];
cx q[0], q[7];
cx q[2], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[7];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[7];
cx q[2], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[8];
u3(0.0, 0.0, 0.04908738521234052) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 6.23409792196725) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
u3(0.0, 0.0, 0.04908738521234052) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 6.18501053675491) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
u3(0.0, 0.0, 0.09817477042468103) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 6.08683576633022) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
u3(0.0, 0.0, 0.19634954084936207) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 5.890486225480862) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.39269908169872414) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 0.7853981633974483) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[0], q[3];
u3(1.5707963267948966, 4.71238898038469, 1.5707963267948966) q[1];
u3(0.0, 0.0, 5.497787143782138) q[3];
cx q[0], q[3];
cx q[2], q[0];
u3(0.0, 0.0, 6.23409792196725) q[0];
u3(10.995574287564276, 0.0, 2.356194490192345) q[3];
cx q[0], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[8];
cx q[2], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[0];
cx q[2], q[7];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[2], q[7];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[2], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
u3(0.0, 0.0, 0.0490873732051035) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[7];
cx q[2], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
cx q[2], q[6];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[2], q[6];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[2], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[6];
cx q[2], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
cx q[2], q[5];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[2], q[5];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[2], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[5];
cx q[2], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
cx q[2], q[4];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[2], q[4];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[2], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[0], q[4];
cx q[2], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[0], q[4];
cx q[2], q[8];
u3(0.0, 0.0, 5.497787143782138) q[4];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[4];
cx q[2], q[8];
cx q[2], q[0];
u3(0.0, 0.0, 6.23409792196725) q[8];
u3(0.0, 0.0, 0.04908738521234052) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[0], q[8];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[8];
cx q[2], q[0];
u3(0.0, 0.0, 6.23409792196725) q[0];
cx q[2], q[7];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[2], q[7];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[2], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
u3(0.0, 0.0, 6.23409795358979) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[7];
cx q[2], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
cx q[2], q[6];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[2], q[6];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[2], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[6];
cx q[2], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
cx q[2], q[5];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[2], q[5];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[2], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[5];
cx q[2], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
cx q[2], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[2], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[2], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[0], q[4];
cx q[2], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[0], q[4];
cx q[2], q[8];
u3(0.0, 0.0, 0.7853981633974483) q[4];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[4];
cx q[2], q[8];
cx q[2], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[8];
u3(0.0, 0.0, 6.23409792196725) q[0];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[0], q[8];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[8];
cx q[2], q[0];
u3(0.0, 0.0, 0.04908738521234052) q[0];
cx q[2], q[7];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[2], q[7];
u3(0.0, 0.0, 6.23409792196725) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[2], q[0];
u3(0.0, 0.0, 6.18501053675491) q[0];
u3(0.0, 0.0, 0.04908738521234052) q[8];
cx q[0], q[7];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[7];
cx q[2], q[0];
u3(0.0, 0.0, 0.09817477042468103) q[0];
cx q[2], q[6];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[2], q[6];
u3(0.0, 0.0, 6.18501053675491) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[2], q[0];
u3(0.0, 0.0, 6.08683576633022) q[0];
u3(0.0, 0.0, 0.09817477042468103) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[6];
cx q[2], q[0];
u3(0.0, 0.0, 0.19634954084936207) q[0];
cx q[2], q[5];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[2], q[5];
u3(0.0, 0.0, 6.08683576633022) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[2], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
u3(0.0, 0.0, 0.19634954084936207) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[5];
cx q[2], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
cx q[2], q[4];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[2], q[4];
u3(0.0, 0.0, 5.890486225480862) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[2], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.39269908169872414) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 0.7853981633974483) q[4];
cx q[0], q[4];
cx q[2], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[0], q[4];
u3(1.5707963267948966, 4.71238898038469, 1.5707963267948966) q[2];
u3(0.0, 0.0, 5.497787143782138) q[4];
cx q[0], q[4];
u3(10.995574287564276, 0.0, 2.356194490192345) q[4];

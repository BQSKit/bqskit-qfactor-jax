OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
cx q[1], q[8];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.35588354) q[0];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 5.92730176717959) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 0.71176709) q[0];
u3(0.0, 0.0, 5.92730175358979) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[7];
cx q[1], q[0];
u3(0.0, 0.0, 5.57141821717959) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 1.4235342) q[0];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 4.85965110717959) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 1.276272) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 1.276272) q[0];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 5.00691330717959) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.9817477042468103) q[0];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.30143760293278) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
cx q[1], q[2];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[1], q[2];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[2];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[0], q[2];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[0], q[2];
cx q[1], q[8];
u3(0.0, 0.0, 0.7853981633974483) q[2];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[2];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.35588354) q[8];
u3(0.0, 0.0, 5.92730176717959) q[0];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[0], q[8];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 0.35588354) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 5.57141821717959) q[0];
u3(0.0, 0.0, 0.3558835) q[8];
cx q[0], q[7];
cx q[8], q[9];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[7];
cx q[1], q[0];
u3(0.0, 0.0, 0.71176709) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 4.85965110717959) q[0];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 1.4235342) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 1.276272) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 5.00691330717959) q[0];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 1.276272) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.30143760293278) q[0];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.9817477042468103) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
cx q[1], q[2];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[1], q[2];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[2];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[0], q[2];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[0], q[2];
cx q[1], q[8];
u3(0.0, 0.0, 5.497787143782138) q[2];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[2];
cx q[1], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 5.92730176717959) q[8];
u3(0.0, 0.0, 0.35588354) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[0], q[8];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[8];
cx q[1], q[0];
u3(0.0, 0.0, 5.92730176717959) q[0];
cx q[1], q[7];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[1], q[7];
u3(0.0, 0.0, 0.35588354) q[8];
cx q[0], q[8];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[1], q[0];
u3(0.0, 0.0, 0.71176709) q[0];
u3(0.0, 0.0, 5.92730176717959) q[8];
cx q[0], q[7];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[7];
cx q[1], q[0];
u3(0.0, 0.0, 5.57141821717959) q[0];
cx q[1], q[6];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[1], q[6];
u3(0.0, 0.0, 0.71176709) q[7];
cx q[0], q[7];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[1], q[0];
u3(0.0, 0.0, 1.4235342) q[0];
u3(0.0, 0.0, 5.57141821717959) q[7];
cx q[0], q[6];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[6];
cx q[1], q[0];
u3(0.0, 0.0, 4.85965110717959) q[0];
cx q[1], q[5];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 1.276272) q[5];
cx q[1], q[5];
u3(0.0, 0.0, 1.4235342) q[6];
cx q[0], q[6];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[1], q[0];
u3(0.0, 0.0, 1.276272) q[0];
u3(0.0, 0.0, 4.85965110717959) q[6];
cx q[0], q[5];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[5];
cx q[1], q[0];
u3(0.0, 0.0, 5.00691330717959) q[0];
cx q[1], q[4];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[1], q[4];
u3(0.0, 0.0, 1.276272) q[5];
cx q[0], q[5];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[1], q[0];
u3(0.0, 0.0, 0.9817477042468103) q[0];
u3(0.0, 0.0, 5.00691330717959) q[5];
cx q[0], q[4];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[4];
cx q[1], q[0];
u3(0.0, 0.0, 5.30143760293278) q[0];
cx q[1], q[3];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[1], q[3];
u3(0.0, 0.0, 0.9817477042468103) q[4];
cx q[0], q[4];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[1], q[0];
u3(0.0, 0.0, 0.39269908169872414) q[0];
u3(0.0, 0.0, 5.30143760293278) q[4];
cx q[0], q[3];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[3];
cx q[1], q[0];
u3(0.0, 0.0, 5.890486225480862) q[0];
cx q[1], q[2];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[1], q[2];
u3(0.0, 0.0, 0.39269908169872414) q[3];
cx q[0], q[3];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[1], q[0];
u3(0.0, 0.0, 0.7853981633974483) q[0];
u3(0.0, 0.0, 5.890486225480862) q[3];
cx q[0], q[2];
u3(0.0, 0.0, 5.497787143782138) q[2];
cx q[0], q[2];
cx q[1], q[0];
u3(0.0, 0.0, 5.497787143782138) q[0];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[0], q[2];
u3(0.0, 0.0, 0.7853981633974483) q[2];
cx q[0], q[2];
u3(0.0, 0.0, 5.497787143782138) q[2];

Point(1) = {0, 0, 0, 1.0};
Point(2) = {0.5, 0, 0, 1.0};
Point(3) = {1.0, 0.1, 0, 1.0};
Point(4) = {1.5, 0, 0, 1.0};
Point(5) = {3, 0, 0, 1.0};
Point(6) = {3, 1, 0, 1.0};
Point(7) = {1.5, 1, 0, 1.0};
Point(8) = {1.0, 1, 0, 1.0};
Point(9) = {0.5, 1, 0, 1.0};
Point(10) = {0, 1, 0, 1.0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 1};

Line(11) = {2, 9};
Line(12) = {3, 8};
Line(13) = {4, 7};

Curve Loop(1) = {10, 1, 11, 9};
Plane Surface(1) = {1};
Curve Loop(2) = {-11, 2, 12, 8};
Plane Surface(2) = {2};
Curve Loop(3) = {-12, 3, 13, 7};
Plane Surface(3) = {3};
Curve Loop(4) = {-13, 4, 5, 6};
Plane Surface(4) = {4};

Transfinite Curve {-10, 11, 12, 13, 5} = 31 Using Progression 1;
Transfinite Curve {1, -9} = 11 Using Progression 0.85;
Transfinite Curve {2, -8} = 21 Using Progression 1;
Transfinite Curve {3, -7} = 21 Using Progression 1;
Transfinite Curve {4, -6} = 41 Using Progression 1.01;

Transfinite Surface {1} = {1, 2, 9, 10};
Transfinite Surface {2} = {2, 3, 8, 9};
Transfinite Surface {3} = {3, 4, 7, 8};
Transfinite Surface {4} = {4, 5, 6, 7};

Recombine Surface {1, 2, 3, 4};

Physical Curve("Left") = {10};
Physical Curve("Upstream") = {1};
Physical Curve("Wedge") = {2, 3};
Physical Curve("Downstream") = {4};
Physical Curve("Right") = {5};
Physical Curve("Top") = {6, 7, 8, 9};
Physical Surface("Domain") = {1, 2, 3, 4};

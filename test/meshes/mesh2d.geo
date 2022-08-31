Point(1) = {0, 0, 0};
Point(2) = {2, 0, 0};
Point(3) = {2, 1, 0};
Point(4) = {0, 1, 0};

Point(5) = {0.5, 0.5, 0};
Point(6) = {0.3, 0.5, 0};
Point(7) = {0.5, 0.3, 0};
Point(8) = {0.7, 0.5, 0};
Point(9) = {0.5, 0.7, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(1) = {1, 2};
Transfinite Curve {2, -4} = 6 Using Progression 1;
Transfinite Curve {1, -3} = 11 Using Progression 1;
Recombine Surface {1};

Physical Curve("Bottom") = {1};
Physical Curve("Right") = {2};
Physical Curve("Top") = {3};
Physical Curve("Left") = {4};
Physical Curve("Hole") = {5, 6, 7, 8};
Physical Surface("Domain") = {1};

Point(1) = {0, 0.1, 0};
Point(2) = {0.9, 0.1, 0};
Point(3) = {0.9, 1, 0};
Point(4) = {3, 1, 0};
Point(5) = {3, 0.1, 0};
Point(6) = {1.5, 0.1, 0};
Point(7) = {1.5, 0.3, 0};
Point(8) = {0.9, 0.3, 0};
Point(9) = {0.7, 0.1, 0};

Circle(1) = {1, 2, 3};
Line(2) = {3, 4};
Line(3) = {4, 5};
Line(4) = {5, 6};
Line(5) = {6, 7};
Line(6) = {7, 8};
Circle(7) = {8, 2, 9};
Line(8) = {9, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {1};

Recombine Surface {1};
Extrude {{1, 0, 0}, {0, 0, 0}, Pi/2} {
  Surface{1}; Layers{5}; Recombine;
}

Physical Surface("Body") = {45, 41, 37};
Physical Surface("Interior") = {49, 33};
Physical Surface("Inlet") = {21};
Physical Surface("Wall") = {25};
Physical Surface("Outlet") = {29};
Physical Surface("Left") = {50};
Physical Surface("Right") = {1};
Physical Volume("Domain") = {1};

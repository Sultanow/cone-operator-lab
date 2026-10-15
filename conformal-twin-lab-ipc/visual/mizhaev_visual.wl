(* ::Package:: *)

(* ::Title:: *)
(*Mizhaev's genus-three polyhedron and its twins*)


(* ::Text:: *)
(*Interactive 3D views and publication exports. Open this file in Mathematica (File > Open), keep the data files*)
(*(mizhaev_P.json, mizhaev_K.json, conformal_hcp_level2.json, hp_disk.json) in the same folder and evaluate*)
(*the cells one by one. Rotate any 3D graphic with the mouse; the current view is kept in vp and vv and used by Export.*)


SetDirectory[If[$InputFileName =!= "", DirectoryName[$InputFileName], NotebookDirectory[]]];

(* the colours of the eight nonagons, identical to the figures in the paper *)
palette = {RGBColor[0.306, 0.475, 0.655], RGBColor[0.949, 0.557, 0.169], RGBColor[0.882, 0.341, 0.349],
   RGBColor[0.463, 0.718, 0.698], RGBColor[0.349, 0.631, 0.310], RGBColor[0.929, 0.788, 0.282],
   RGBColor[0.690, 0.478, 0.631], RGBColor[1.000, 0.616, 0.655]};

loadMesh[file_String] := Module[{d = Import[file, "RawJSON"]},
   <|"V" -> d["vertices"], "F" -> d["faces"], "G" -> d["nonagon"]|>];

P = loadMesh["mizhaev_P.json"];   (* 24 integer vertices, 8 nonagons *)
K = loadMesh["mizhaev_K.json"];   (* the same surface cut into 56 triangles *)


(* ::Section:: *)
(*The polyhedron*)


(* triangles coloured by nonagon (correct also for the non-convex nonagons), nonagon boundaries as lines *)
polyhedron[opts___] := Graphics3D[{
    GraphicsComplex[K["V"], {EdgeForm[None],
      Table[{palette[[g]], Polygon[Pick[K["F"], K["G"], g]]}, {g, 8}]}],
    {GrayLevel[0.1], AbsoluteThickness[1.6], Line[P["V"][[Append[#, First[#]]]]] & /@ P["F"]}},
   opts, Boxed -> False, Lighting -> "Neutral", SphericalRegion -> True, ImageSize -> 700];

labels = Table[Text[Style[i, 11, Bold], 1.08 P["V"][[i]]], {i, Length[P["V"]]}];
axis = {Dashed, GrayLevel[0.5], Line[{{0, 0, -330}, {0, 0, 330}}]};

vp = {1.3, -2.4, 1.6}; vv = {0, 0, 1};
polyhedron[ViewPoint -> Dynamic[vp], ViewVertical -> Dynamic[vv]]


(* with vertex numbers (Mizhaev's labels) and the axis of the four-fold symmetry *)
Show[polyhedron[ViewPoint -> Dynamic[vp], ViewVertical -> Dynamic[vv]], Graphics3D[{labels, axis}]]


(* export the view you have rotated to *)
Export["mizhaev_P.png", polyhedron[ViewPoint -> vp, ViewVertical -> vv], ImageResolution -> 300];
Export["mizhaev_P.pdf", polyhedron[ViewPoint -> vp, ViewVertical -> vv]];


(* a turntable animation, e.g. for slides *)
Export["mizhaev_P_turntable.gif",
  Table[polyhedron[ViewPoint -> {3 Cos[t], 3 Sin[t], 1.4}, ImageSize -> 500], {t, 0, 2 Pi - Pi/36, Pi/36}],
  "DisplayDurations" -> 0.06, "AnimationRepetitions" -> Infinity];


(* ::Section:: *)
(*A smooth surface in space, conformal to the circle-packing twin (cluster run, level 2)*)


surface[mesh_, opts___] := Graphics3D[
   GraphicsComplex[mesh["V"], {EdgeForm[{GrayLevel[0.35], AbsoluteThickness[0.15]}],
     Table[{palette[[g]], Polygon[Pick[mesh["F"], mesh["G"], g]]}, {g, 8}]}],
   opts, Boxed -> False, Lighting -> "Neutral", SphericalRegion -> True, ImageSize -> 700];

S = loadMesh["conformal_hcp_level2.json"];
surface[S, ViewPoint -> Dynamic[vp], ViewVertical -> Dynamic[vv]]


GraphicsRow[{polyhedron[ViewPoint -> vp, ViewVertical -> vv], surface[S, ViewPoint -> vp, ViewVertical -> vv]},
  ImageSize -> 1000]


(* any other checkpoint: python scripts/export_obj.py RESULT_L4.npz OUTDIR, then loadMesh["OUTDIR/RESULT_L4.json"] *)


(* ::Section:: *)
(*The canonical twin h_P in the Poincare disk*)


hp = Import["hp_disk.json", "RawJSON"];
coneColor[a_] := ColorData["TemperatureMap"][Rescale[a, {0, 720}]];

canonicalTwin = Graphics[{
    {GrayLevel[0.96], EdgeForm[{GrayLevel[0.2], AbsoluteThickness[1]}], Disk[{0, 0}, 1]},
    {EdgeForm[None], Opacity[0.65], palette[[#["nonagon"]]], Polygon[#["xy"]]} & /@ hp["triangles"],
    {GrayLevel[0.1], AbsoluteThickness[0.9], Line /@ hp["edges"]},
    {EdgeForm[{Black, AbsoluteThickness[0.5]}], {coneColor[#["angle"]], Disk[#["xy"], 0.013]} & /@ hp["cone_points"]}},
   PlotRange -> 1.02, ImageSize -> 600];
Legended[canonicalTwin,
  BarLegend[{"TemperatureMap", {0, 720}}, LegendLabel -> "cone angle on P (degrees)"]]


Export["canonical_twin.pdf", canonicalTwin];

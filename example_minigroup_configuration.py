import numpy as np
example_minigroup_configuration = np.array([ 455,   74,  500,  447,  112,  705,  964,   13,  117,  418,  820,
         20,  506,  800,  116,  811,  324,  135,  672,  247,  679,  762,
        509,  299,  177,  358,  219,  415,  353, 1030,  892, 1041,  304,
        123,  994,  799,  464,  571,  546,  549,  623,  161,  692,  936,
        851,  485,  745,  869, 1038,  227,  327,  431,  192,  901,  575,
        297,  963,  786,  377,  918,  696,  697,  648,  594,  183,  826,
         64,   72,  534,   86, 1047,  413,  617,  367,  574,  126,  717,
        580,  381,  780,  323,  221,  163,  379,    4,  229, 1022,  674,
        241,  553,  972,  888,  276,  980,  278,  993,  794,  682,  852,
        550,  588,  176,  978,  979,  257,  721,  199,  611,  238,  508,
        416,  872,  822,  775,   80,  803,  708,  977,  801,  217,  489,
        988,  646,  205,  640,   30,   41,  160,  690,   36, 1049,  523,
        764,   91,  995,  384,  308,  957,  931,  691,  281,  359,   34,
        168,  908,  556,  284,   62,  680,  374,  132,  478,  393,  567,
        349,  371,  663,  982,  806,  661,  211,  142,  348,  633,  119,
        597,  370,  390,  668,  783,  184, 1000,  698,  125,  714, 1028,
        410,  636,   77,  237,  830,   18,   63,  939,  585,  318,  243,
        625,  200, 1011,  397,  175,  488,  134,   22,  484,   15,  747,
        409,  437,  341,  222,  678,  687,  796,  366,  446,  810,  407,
        273,  713,  181,  270,  492,  798,  774,  772,  274,  114,  876,
        539,  608,  145,  283,  642,  245,  350,  423,  545,  832,  104,
        468, 1033,  723,  440,  333,  303, 1029,   96,   47,  958,  996,
        552,  232,  650,  946,  256,    8,  947,  903,  940,  768,  472,
        307,  529,  251,  339,  101,  398,  133,  373,  150,  945,  331,
        807,  164,  771,  583,  595, 1034,  711,  858,  194,  613,  951,
        849,  761,  834, 1002,  388,  143,  907,  153,  742,  699,  230,
        259,  424,  629,  210,  868,   66,  314,  924,  725,  531,  122,
        766,  467, 1019,   55,  652,  828,  411,  896,  887,  802,  522,
        675,  906,  188,  850,   56,  683,  917,  827,  565,  405,  255,
        809,   93,  981,   69,  782, 1045,  818,   43,  955,  236,  520,
        334,  893,  305,   50,  704,  601,  812,  362,   65,  938,  570,
        673, 1036,   26,  658,  943,  976,   44,  933,  130,  921,  788,
        481,  351,  471,  719,  912,  560,  452,  967,  969,  879,  443,
       1003,  990,  141,  909,  754,  214, 1024,  859,  540,  784,  627,
        986,  566,  874,  521, 1016,  285,  340,  665,   94,  791,  380,
        429,  757, 1031,   48,  103,  817,  118,  395,    1,   42,  855,
        466, 1023,  357,  474,  657,  436,   32,   82,  935,  182,  309,
        621,  515,  234,  564,  207,  561,  870,  968,  760,  694,  226,
        100,  835,  857,   87, 1043,  544,  911,  877,  749,  108,  860,
        902,   45,  953,  242,  600,  216,  971,  558,  952,  654, 1009,
        974,  729,  338,  432,  620,  900,  365,  873,  638,  166,   10,
         73,  904,  441,  427,  282,  987,  839,  603, 1042,  950,  137,
       1048,  841,  667,  330,  322,  250,  201,   38,  644,  985,  751,
        248,  825,  962,   58,  966,  325,  344,  290,  853,  178,  606,
        681,   92, 1015,  311,  741,  628,  490,  703,  115,  641, 1040,
        392,  146,  897,  587,  497,   16,  864,  233,  525,  582,  758,
        942,  722,  836,  172,  159,  109,  280,  557,   12,   33,  426,
        808,  551,  824,  709,  240,  499,  272,  401,  920,  190,  635,
        769,  614,  301,  684,  975,  579,  894,  315,  428,  253,  677,
        235,   11,  444,  169,  701,    2,  715,  584,  804,  989,  312,
        394,  647,   21,  185,  195,  224,  291,  631,  469,  162,  871,
        671,  136, 1046,   60,  476,  203,  439,  399,  519,  563,  948,
        496,  298,   83,  450,  310,  347,  171,  220,  837,  750,  591,
        562,  479,  456,  624,  458,  738,  504,  535,  387,  716,  913,
        537,  511,  928, 1013,  111,  792,  732,  328,  296,  910,  884,
         98,  336, 1001,  124,  914,  821,  404,  197,  706,  202,  457,
        420,   25,  502,   24,  685,  727,  881,  720,  414,  223,  482,
        724,  612,  317, 1037,  286,  461,  756,  483,  815,   88,  470,
         49,  152,  730,  215,  844,  442,  885,  487,  524,  189,  843,
        734,  707,  785,  300,  422,  419,  517,  666,  231,  313,  386,
        863,  622,  590,  174,  592,  332,  445,   51,  396,  363,  604,
        530,  449,  369,  403,  475,  391, 1008,  187,  462,  252, 1007,
        225,  279,  128,  147, 1027,  239,  361,  140,  793,  664,  656,
       1012,  889,  941,  516,  541,  505,  676,   99,   39,  773,  543,
        121,  503,  630,  777,  501,  289,   59,  120,  740,  867,    0,
        899,  421,  973,  355,  155,  402,  923,  637,  602,  491,  735,
        831,  158,  572,    6,  662,  218,   68, 1032,  294,  891,  833,
        316,  847,  829,  639,  660, 1020,  262, 1010,    3,  649,  113,
        507,   67,  930,  385,  573,  568,   37,  726,  767,  438,  170,
        959,  343,  686,  718,  532,  514,  542,  167,  626,  752,  593,
         97,  151,  480, 1025,   46,  212,  267,  932,  998,    9,  383,
        265,  651,  960,  321,  733,  326,  295,  753,  634,  856,  345,
        258,  375,  193,  845,  688,  105,  453,  645,  106, 1017,   81,
         70,  408,  495, 1006,  865, 1018,  425,  823,  581,  206,  129,
        610,  589,  157,  569,  486,  578,  417,   14,   95,  498, 1050,
        494,  451,  547,  927,  406,  700,  460,  743,  154,   57,  605,
        746,  586,  895,  787,  862,  854,  536,  288,  356,  776,  548,
        434,  127,  814,  477,  883,   84,  882,  110,  616,  329,  555,
        748,  838,  412,   90,  277,  618,  269,  144,  346,  337,  186,
        510,  789,  538,  165,  916,  797,  260,  670,  702,  204,  513,
        878,  926,  736,  770,  198,  790,  559,  778,  179,  292,  693,
        759,  949, 1014,  695,   75,  463,  731,  389,  554,  342,  254,
        246,  319,  191,  970,  866, 1044,  459,   71,  577,  433,  352,
        196,  261,  213,  173,  302,  376,  512,  264,  805,  102,   79,
        643,  596,   78, 1021,  956,    5,  107,  763,  890,  875, 1039,
        599,  898,  306,  368,  861,  886, 1026,   85,  934,  228,  465,
        293,  180,  335,  880,  615,  271,  473,  576,   19,    7,  266,
        991,   31,  138, 1051,  795,  372,  607,  689,  744,  712,  244,
        382,  518,  944,  354,  378,  598,  983,  268,  992,  209,  779,
         23,  905,  937,  275,  609,  156,   52,  139,  997,  263,  929,
         17,  954,  999,  364,   35,  922,  737,  360, 1004,  320,  816,
         29,  249,   27,  765,  493,  287,  528,  728,  755,   76,  526,
        148,  435,  961,  454,  925,  533,  781,   40,  659, 1005,  984,
        819,  848,  915,  448,  840,  846,  131,  632,  739,  400,  669,
        813,  655,  919,  527,   61,   28,  842,   89, 1035,   53,  619,
        653,  710,   54,  208,  430,  149,  965])

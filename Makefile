extract_data: extract_data.cxx 
	g++ -o extract_data extract_data.cxx -lz `root-config --cflags --libs`


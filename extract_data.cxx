#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
#include <map>

unsigned tpgRawDataPacketWords(unsigned ntc) {
  unsigned nWords;
  assert(ntc<=48);
  
  if(ntc==0) { // No hits
    nWords=1;    
  } else if(ntc<8) { // Low occupancy
    nWords=(16 + 3 + 13*ntc + 15)/16;
  } else { // High occupancy
    nWords=(16 + 48 + 7*ntc + 15)/16;
  }
  
  return nWords;
}
TH3D * convert_tcs_to_words( TH3D * hist ){
  
  TH3D * words = (TH3D*) hist->Clone( TString(hist->GetName()) + "_words" );
  
  for ( int i = 1; i < words->GetNbinsX()+1; i++){
    for ( int j = 1; j < words->GetNbinsY()+1; j++){
      for ( int k = 1; k < words->GetNbinsZ()+1; k++){
	
	unsigned ntcs = words->GetBinContent(i,j,k);
	unsigned nwords = tpgRawDataPacketWords(ntcs);

	if (nwords < 0.5){
	  std::cout << "nwords = " << nwords << std::endl;
	}
	words->SetBinContent(i,j,k,nwords);
	
      }
    }
  }
  
  return words;
  
}

unsigned uvMapping(unsigned layer, std::pair<int,int> &uv) {
  unsigned sector(0);
  int offset;
  
  if(layer<=28) { // CE-E    
    if(uv.first>0 && uv.second>=0) return sector;
    
    offset=0;
    if(uv.first>=uv.second && uv.second<0) sector=2;
    else sector=1;
    
  } else if((layer%2)==1) { // CE-H Odd
    if(uv.first>=0 && uv.second>=0) return sector;
    
    offset=-1;    
    if(uv.first>uv.second && uv.second<0) sector=2;
    else sector=1;
    
  } else { // CE-H Even
    if(uv.first>=1 && uv.second>=1) return sector;
    
    offset=1;
    if(uv.first>=uv.second && uv.second<1) sector=2;
    else sector=1;
  }
  
  int up,vp;
  
  if(sector==1) {
    up=uv.second-uv.first;
    vp=-uv.first+offset;    
    
  } else {
    up=-uv.second+offset;
    vp=uv.first-uv.second+offset;
  }
  
  uv.first=up;
  uv.second=vp;
  return sector;
}

//Rotate and convert from cell to tile numbering
unsigned etaphiMapping(unsigned layer, std::pair<int,int> &etaphi) {
  unsigned sector(0);
  
  if (etaphi.second <= 48){
    sector = 0;
  }
  else if (etaphi.second > 48 && etaphi.second <= 96){
    sector = 1;
  }
  else {
    sector = 2;
  }
  
  int ep;
  int pp;
  
  if(sector==1) {
    pp=etaphi.second-48;
  } else if(sector==2) {
    pp=etaphi.second-96;
  }
  else{
    pp = etaphi.second;
  }
  
  pp = (pp-1)/4; //Phi index 1-12
  
  if ( etaphi.first <= 3 ){
    ep = 0;
  }
  else if ( etaphi.first <= 9 ){
    ep = 1;
  }
  else if ( etaphi.first <= 13 ){
    ep = 2;
  }
  else if ( etaphi.first <= 17 ){
    ep = 3;
  }
  else{
    ep = 4;
  }
  
  etaphi.first=ep;
  etaphi.second=pp;
  
  
  return sector;
}

std::pair<float,float> getEtaPhi(float x, float y, float z){

  float r = std::sqrt( x*x + y*y  );
  float phi = std::atan2(y,x);
  float eta = -std::log(std::tan(std::atan2( r,z)/2));
  return std::make_pair(eta,phi);

}
std::pair<float,float> getROverZPhi(float x, float y, float z){

  float r = std::sqrt( x*x + y*y  );
  float phi = std::atan2(y,x);

  phi = phi + M_PI;
  if ( phi < (2*M_PI/3) )
    phi = phi;
  else if ( phi < (4*M_PI/3) )
    phi = phi-(2*M_PI/3);
  else
    phi = phi-(4*M_PI/3);

  return std::make_pair(r/z,phi);

}


int main(){
  TH1::SetDefaultSumw2();
  TH1::AddDirectory(kFALSE);
  //TFile * file = new TFile("data/PU200-V11-TTBAR-2.root","READ");
  //TFile * file = new TFile("data/PU200-QG.root","READ");
  TFile * file = new TFile("data/PU200-3.root","READ");
  TTree * tree = (TTree*)file->Get("HGCalTriggerNtuple");
  
    // Declaration of leaf types
  std::vector<int>     *tc_layer = 0;
  std::vector<int>     *tc_waferu = 0;
  std::vector<int>     *tc_waferv = 0;
  std::vector<int>     *tc_cellu = 0;
  std::vector<int>     *tc_cellv = 0;
  std::vector<int>     *tc_panel_number = 0;
  std::vector<int>     *tc_panel_sector = 0;
  std::vector<int>     *tc_zside = 0;
  std::vector<float>     *tc_x = 0;
  std::vector<float>     *tc_y = 0;
  std::vector<float>     *tc_z = 0;
  // List of branches
  TBranch        *b_tc_layer = 0;   
  TBranch        *b_tc_waferu = 0;   
  TBranch        *b_tc_waferv = 0;
  TBranch        *b_tc_cellu = 0;   
  TBranch        *b_tc_cellv = 0;
  TBranch        *b_tc_panel_number = 0;   
  TBranch        *b_tc_panel_layer = 0;
  TBranch        *b_tc_zside = 0;
  TBranch        *b_tc_x = 0;
  TBranch        *b_tc_y = 0;
  TBranch        *b_tc_z = 0;   
    
  tree->SetBranchAddress("tc_layer" , &tc_layer , &b_tc_layer);
  tree->SetBranchAddress("tc_waferu", &tc_waferu, &b_tc_waferu);
  tree->SetBranchAddress("tc_waferv", &tc_waferv, &b_tc_waferv);
  tree->SetBranchAddress("tc_cellu", &tc_cellu, &b_tc_cellu);
  tree->SetBranchAddress("tc_cellv", &tc_cellv, &b_tc_cellv);
  tree->SetBranchAddress("tc_zside", &tc_zside, &b_tc_zside);

  tree->SetBranchAddress("tc_x", &tc_x, &b_tc_x);
  tree->SetBranchAddress("tc_y", &tc_y, &b_tc_y);
  tree->SetBranchAddress("tc_z", &tc_z, &b_tc_z);
  
  std::vector<TH3D*> per_event_plus(3);
  std::vector<TH3D*> per_event_minus(3);
  
  std::vector<TH3D*> per_event_plus_scin(3);
  std::vector<TH3D*> per_event_minus_scin(3);
  
  for (int i = 0;i<per_event_plus.size();i++ ){
    per_event_plus.at(i) = new TH3D(TString("per_event_plus_hist" + std::to_string(i)),"",15,-0.5,14.5,15,-0.5,14.5,52,0.5,52.5) ;
    per_event_minus.at(i) = new TH3D(TString("per_event_minus_hist" + std::to_string(i)),"",15,-0.5,14.5,15,-0.5,14.5,52,0.5,52.5) ;
    per_event_plus_scin.at(i) = new TH3D(TString("per_event_plus_hist_scin" + std::to_string(i)),"",5,-0.5,4.5,12,-0.5,11.5,52,0.5,52.5) ;
    per_event_minus_scin.at(i) = new TH3D(TString("per_event_minus_hist_scin" + std::to_string(i)),"",5,-0.5,4.5,12,-0.5,11.5,52,0.5,52.5) ;
  }
  
    TH3D * out_words = new TH3D("out_words_hist","",15,-0.5,14.5,15,-0.5,14.5,52,0.5,52.5);
    TH3D * out_tcs = new TH3D("out_tcs_hist","",15,-0.5,14.5,15,-0.5,14.5,52,0.5,52.5);

    TH3D * out_words_scin = new TH3D("out_words_hist_scin","",5,-0.5,4.5,12,-0.5,11.5,52,0.5,52.5);
    TH3D * out_tcs_scin = new TH3D("out_tcs_hist_scin","",5,-0.5,4.5,12,-0.5,11.5,52,0.5,52.5);

    //R/Z Histograms

    TH1D * ROverZ_Inclusive = new TH1D("ROverZ_Inclusive","",42,0.076,0.58);
    TH1D * ROverZ_Inclusive_Phi60 = new TH1D("ROverZ_Inclusive_Phi60","",42,0.076,0.58);
    std::map<std::tuple<int,int,int,int>,TH1D*> ROverZ_per_module;
    std::map<std::tuple<int,int,int,int>,TH1D*> ROverZ_per_module_Phi60;
    //Create one for each module (silicon at first)
    for ( int i = 0; i < 15; i++){//u
      for ( int j = 0; j < 15; j++){//v
	for ( int k = 1; k < 53; k++){//layer

	  if ( k < 28 && k%2 == 0 ) continue;
	  ROverZ_per_module[std::make_tuple(0,i,j,k)] = new TH1D( ("ROverZ_silicon_" + std::to_string(i) + "_" +  std::to_string(j) +"_"+ std::to_string(k)).c_str(),"",42,0.076,0.58);
	  ROverZ_per_module_Phi60[std::make_tuple(0,i,j,k)] = new TH1D( ("ROverZ_Phi60_silicon_" + std::to_string(i) + "_" +  std::to_string(j) +"_"+ std::to_string(k)).c_str(),"",42,0.076,0.58);
	  
	}
      }
    }
	  
    for ( int i = 0; i < 5; i++){
      for ( int j = 0; j < 12; j++){
	for ( int k = 37; k < 53; k++){

	  ROverZ_per_module[std::make_tuple(1,i,j,k)] = new TH1D( ("ROverZ_scintillator_" + std::to_string(i) + "_" +  std::to_string(j) +"_"+ std::to_string(k)).c_str(),"",42,0.076,0.58);
	  ROverZ_per_module_Phi60[std::make_tuple(1,i,j,k)] = new TH1D( ("ROverZ_Phi60_scintillator_" + std::to_string(i) + "_" +  std::to_string(j) +"_"+ std::to_string(k)).c_str(),"",42,0.076,0.58);

	}
      }
    }

    std::ofstream fpaul;
    fpaul.open ("words_all_v11.txt");
    
    
    Int_t nentries = (Int_t)tree->GetEntries();
    Int_t nentries_looped = 0;
    Long64_t nb = 0;
    for (Long64_t jentry=0; jentry<nentries;jentry++) {
      nb = tree->GetEntry(jentry);   
      if (jentry % 100 == 0) std::cout << jentry << " / " << nentries << std::endl;;
      //      if (jentry > 100 )break;

      for (int j = 0;j<tc_waferu->size();j++){

	int u = tc_waferu->at(j);
	int v = tc_waferv->at(j);

	//Fill Eta Histograms	
	std::pair<float,float> eta_phi = getROverZPhi(tc_x->at(j),tc_y->at(j),tc_z->at(j));
	
	ROverZ_Inclusive->Fill(std::abs(eta_phi.first));
	if ( eta_phi.second < M_PI/3 )
	  ROverZ_Inclusive_Phi60->Fill(std::abs(eta_phi.first));

	
	if ( u > -990 ){//Silicon
	  std::pair<int,int> uv = std::make_pair(u,v);
	  unsigned sector = uvMapping(tc_layer->at(j),uv);
	  
	  if ( tc_zside->at(j) > 0 ){
	    per_event_plus.at(sector)->Fill(uv.first , uv.second, tc_layer->at(j) );
	  }
	  else if ( tc_zside->at(j) < 0 ){
	    per_event_minus.at(sector)->Fill(uv.first , uv.second, tc_layer->at(j) );
	  }


	  ROverZ_per_module[std::make_tuple(0,uv.first,uv.second,tc_layer->at(j))]->Fill(std::abs(eta_phi.first));
	  if ( eta_phi.second < M_PI/3 )
	    ROverZ_per_module_Phi60[std::make_tuple(0,uv.first,uv.second,tc_layer->at(j))]->Fill(std::abs(eta_phi.first));
	      
	  
	}
	else{
	  int eta = tc_cellu->at(j);
	  int phi = tc_cellv->at(j);
	
	  std::pair<int,int> etaphi = std::make_pair(eta,phi);
	  unsigned sector = etaphiMapping(tc_layer->at(j),etaphi);
	  if ( tc_zside->at(j) > 0 ){
	    per_event_plus_scin.at(sector)->Fill(etaphi.first , etaphi.second, tc_layer->at(j) );
	    //	    std::cout << "PLus s ector  = " << sector << std::endl;
	  }
	  else if ( tc_zside->at(j) < 0 ){
	    per_event_minus_scin.at(sector)->Fill(etaphi.first , etaphi.second, tc_layer->at(j) );
	    //	    std::cout << "Minus s ector  = " << sector << std::endl;
	  }


	  ROverZ_per_module[std::make_tuple(1,etaphi.first,etaphi.second,tc_layer->at(j))]->Fill(std::abs(eta_phi.first));
	  if ( eta_phi.second < M_PI/3 )
	    ROverZ_per_module_Phi60[std::make_tuple(1,etaphi.first,etaphi.second,tc_layer->at(j))]->Fill(std::abs(eta_phi.first));



	}

      

	
      }

      std::vector<TH3D*> words_plus;
      std::vector<TH3D*> words_minus;
      std::vector<TH3D*> words_plus_scin;
      std::vector<TH3D*> words_minus_scin;
      //Convert number of trigger cells, to words
      for (int i = 0;i<per_event_plus.size();i++ ){
	words_plus.push_back(convert_tcs_to_words(per_event_plus.at(i)));
	words_minus.push_back(convert_tcs_to_words(per_event_minus.at(i)));
	words_plus_scin.push_back(convert_tcs_to_words(per_event_plus_scin.at(i)));
	words_minus_scin.push_back(convert_tcs_to_words(per_event_minus_scin.at(i)));	
      }

      //Do something for Paul

      // for (int z = 0;z<words_plus_scin.size();z++ ){

      // 	for ( int i = 0; i < 5; i++){
      // 	  for ( int j = 0; j < 12; j++){
      // 	    for ( int k = 37; k < 53; k++){	
      // 	      fpaul << words_plus_scin.at(z)->GetBinContent(words_plus_scin.at(z)->FindBin(i,j,k)) << " ";
      // 	    }
      // 	  }
      // 	}

      // 	fpaul <<  std::endl;
	
      // 	for ( int i = 0; i < 5; i++){
      // 	  for ( int j = 0; j < 12; j++){
      // 	    for ( int k = 37; k < 53; k++){
      // 	      fpaul << words_minus_scin.at(z)->GetBinContent(words_minus_scin.at(z)->FindBin(i,j,k)) << " ";
      // 	    }
      // 	  }
      // 	}
	
      // 	fpaul <<  std::endl;
	
      // }
      

      for (int z = 0;z<words_plus.size();z++ ){

      	for ( int i = 0; i < 15; i++){
      	  for ( int j = 0; j < 15; j++){
      	    for ( int k = 1; k < 53; k++){	
      	      fpaul << words_plus.at(z)->GetBinContent(words_plus.at(z)->FindBin(i,j,k)) << " ";
      	    }
      	  }
      	}

      	fpaul <<  std::endl;
	
      	for ( int i = 0; i < 15; i++){
      	  for ( int j = 0; j < 15; j++){
      	    for ( int k = 1; k < 53; k++){
      	      fpaul << words_minus.at(z)->GetBinContent(words_minus.at(z)->FindBin(i,j,k)) << " ";
      	    }
      	  }
      	}
	
      	fpaul <<  std::endl;
	
      }
      
      
      //Add plus and minus sides and all rotated histograms together
      for (int i = 0;i<per_event_plus.size();i++ ){
	out_tcs->Add(per_event_plus.at(i));
	out_tcs->Add(per_event_minus.at(i));
	out_words->Add(words_plus.at(i));
	out_words->Add(words_minus.at(i));
	out_tcs_scin->Add(per_event_plus_scin.at(i));
	out_tcs_scin->Add(per_event_minus_scin.at(i));
	out_words_scin->Add(words_plus_scin.at(i));
	out_words_scin->Add(words_minus_scin.at(i));
      }
      for (int i = 0;i<per_event_plus.size();i++ ){
	per_event_plus.at(i)->Reset();
	per_event_minus.at(i)->Reset();
	words_plus.at(i)->Delete();
	words_minus.at(i)->Delete();
	per_event_plus_scin.at(i)->Reset();
	per_event_minus_scin.at(i)->Reset();
	words_plus_scin.at(i)->Delete();
	words_minus_scin.at(i)->Delete();
      }
      nentries_looped++;
    }

    fpaul.close();


    out_tcs->Scale(1./double(nentries_looped*6.));
    out_words->Scale(1./double(nentries_looped*6.));
    out_tcs_scin->Scale(1./double(nentries_looped*6.));
    out_words_scin->Scale(1./double(nentries_looped*6.));


    //Save Eta Histograms	
    TFile * file_out = new TFile("ROverZHistograms.root","RECREATE");
    file_out->cd();
    ROverZ_Inclusive->Write();
    ROverZ_Inclusive_Phi60->Write();

    for (auto& x: ROverZ_per_module) {
      x.second->Write();
    }
    for (auto& x: ROverZ_per_module_Phi60) {
      x.second->Write();
    }

    file_out->Close();
    //out_words->SaveAs("hist.root");
    //Create output csv
    std::ofstream fout;
    fout.open ("average_tcs_sil.csv");

    for ( int i = 0; i < 15; i++){
      for ( int j = 0; j < 15; j++){
	for ( int k = 1; k < 53; k++){

	  double ntcs = out_tcs->GetBinContent(out_tcs->FindBin(i,j,k));
	  double words = out_words->GetBinContent(out_words->FindBin(i,j,k));
	  fout << i << "," << j << "," << k << "," << ntcs << "," << words << "\n";
	}
      }
    }

    fout.close();

    fout.open ("average_tcs_scin.csv");
    for ( int i = 0; i < 5; i++){
      for ( int j = 0; j < 12; j++){
	for ( int k = 37; k < 53; k++){

	  double ntcs = out_tcs_scin->GetBinContent(out_tcs_scin->FindBin(i,j,k));
	  double words = out_words_scin->GetBinContent(out_words_scin->FindBin(i,j,k));
	  fout << i << "," << j << "," << k << "," << ntcs << "," << words << "\n";
	}
      }
    }

    fout.close();
    
    out_tcs->Delete();
    out_words->Delete();
    file->Close();
    
    return 0;
}

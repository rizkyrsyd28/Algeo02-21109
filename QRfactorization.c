#include <Rcpp.h>
#include <stdio.h>

// [[Rcpp::export]]

List myQRCpp(NumericMatrix A) {
  int a = A.rows();
  int b = A.cols();
  NumericMatrix U(a,b);
  NumericMatrix Q(a,b);
  NumericMatrix R(a,b);
  NumericMatrix::Column Ucol1 = U(_ , 0);
  NumericMatrix::Column Acol1 = A(_ , 0);
  
  Ucol1 = Acol1;
  
  for(int i = 1; i < b; i++){
    NumericMatrix::Column Ucol = U(_ , i);
    NumericMatrix::Column Acol = A(_ , i);
    NumericVector subt(a);
    int j = 0;
    while(j < i){
      NumericVector uj = U(_ , j);
      NumericVector ai = A(_ , i);
      subt = subt + projC(uj, ai);
      j++;
    }
    Ucol = Acol - subt;
  }

  for(int i = 0; i < b; i++){  // Cari Q[:,i]
    NumericMatrix::Column ui = U(_ , i);
    NumericMatrix::Column qi = Q(_ , i);
    
    double sum2_ui = 0;
    for(int j = 0; j < a; j++){
      sum2_ui = sum2_ui + ui[j]*ui[j];
    }
    qi = ui/sqrt(sum2_ui);
  }
  
  List L = List::create(Named("Q") = Q , _["U"] = U);
  return L;
}

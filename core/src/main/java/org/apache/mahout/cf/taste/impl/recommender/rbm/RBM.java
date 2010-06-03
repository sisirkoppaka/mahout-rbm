/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.impl.recommender.rbm;

import java.util.List;
import java.lang.Math;

import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.cf.taste.common.TasteException;

import org.apache.mahout.math.jet.random.engine.DRand;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A pure RBM algorithm. */
public class RBM {
	
	/** Default optimum constants for 100 hidden variables on the Netflix dataset. */
	private int totalFeatures = 100;
	private int softmax = 5;
	private double epsilonw = 0.001; /** Learning rate for weights */
	private double epsilonvb = 0.008; /** Learning rate for biases of visible units */
	private double epsilonhb = 0.0006; /** Learning rate for biases of hidden units */
	private double weightCost = 0.0001;
	private double momentum = 0.8;
	private double finalMomentum = 0.9;
	  
	private final double e = 0.00002; /* Stop condition */
	
	private int numItems;
	private int numUsers;
	
	private double[][][] vishid;
	private double[][] visbiases;
	private double[] hidbiases;
	private double[][][] CDpos;
	private double[][][] CDneg;
	private double[][][] CDinc;
	private double[][] Dij;
	private double[][] DIJinc;

	private double[] poshidprobs;
	private char[]   poshidstates; 
	private char[]   curposhidstates; 
	private double[] poshidact;
	private double[] neghidact;
	private double[] neghidprobs;
	private char[]   neghidstates; 
	private double[] hidbiasinc;

	private double[][] nvp2;
	private double[][] negvisprobs;
	private char[]   negvissoftmax; 
	private double[][] posvisact;
	private double[][] negvisact;
	private double[][] visbiasinc;

	private int[] moviercount;
	private int[] moviecount;
	private int[] movieseencount;
	
	private int[][] useridx; //TODO: Replace
	private int[] userent; //TODO: Replace
	
	public RBM(int totalFeatures, int softmax, double epsilonw, double epsilonvb, double epsilonhb, double weightCost, double momentum, double finalMomentum) {
		this.totalFeatures = totalFeatures;
		this.softmax = softmax;
		this.epsilonw = epsilonw;
		this.epsilonvb = epsilonvb;
		this.epsilonhb = epsilonhb;
		this.weightCost = weightCost;
		this.momentum = momentum;
		this.finalMomentum = finalMomentum;
		
		/** Bring data structures to life */
		vishid = new double[numItems][softmax][totalFeatures];
		visbiases = new double[numItems][softmax];
		hidbiases = new double[totalFeatures];
		CDpos = new double[numItems][softmax][totalFeatures];
		CDneg = new double[numItems][softmax][totalFeatures];
		CDinc = new double[numItems][softmax][totalFeatures];
		Dij = new double[numItems][totalFeatures];
		DIJinc = new double[numItems][totalFeatures];

		poshidprobs = new double[totalFeatures];
		poshidstates = new char[totalFeatures]; 
		curposhidstates = new char[totalFeatures]; 
		poshidact = new double[totalFeatures];
		neghidact = new double[totalFeatures];
		neghidprobs = new double[totalFeatures];
		neghidstates = new char[totalFeatures]; 
		hidbiasinc = new double[totalFeatures];

		nvp2 = new double[numItems][softmax];
		negvisprobs = new double[numItems][softmax];
		negvissoftmax = new char[numItems]; 
		posvisact = new double[numItems][softmax];
		negvisact = new double[numItems][softmax];
		visbiasinc = new double[numItems][softmax];

		moviercount = new int[softmax*numItems];
		moviecount = new int[numItems];
		movieseencount = new int[numItems];
		
		
	}
	
	private void zero(int[] arraySet, int i) {
		int m;
		
		for(m=0;m<i;m++) {
			arraySet[m]=0;
		}
	}
	
	private void zero(double[] arraySet, int i) {
		int m;
		
		for(m=0;m<i;m++) {
			arraySet[m]=0;
		}
	}
	
	private void zero(double[][] arraySet, int i, int j) {
		int m,n;
		
		for(m=0;m<i;m++) {
			for(n=0;n<j;n++) {
				arraySet[m][n]=0;
			}
		}
	}
	
	private void zero(double[][][] arraySet, int i, int j, int k) {
		int m,n,o;
		
		for(m=0;m<i;m++) {
			for(n=0;n<j;n++) {
				for(o=0;o<k;o++) {
					arraySet[m][n][o]=0;
				}
			}
		}
	}

	public void initScore() {
	    int i,u,m,j,n;
	    int base0, d0;
	    
	    for (m=0; m<numItems; m++) {
	    	for(n=0;n<5;n++) {
	    		moviercount[m*softmax+n] = 0;
	    	}
	    }
	    
	    for(u=0;u<numUsers;u++) {
	        base0=useridx[u][0];
	        d0=untrain(u);

	        // For all rated movies
	        for(j=0;j<d0;j++) {
	            int m=userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	            int r=(userent[base0+j]>>USER_LMOVIEMASK)&7; //TODO: Replace
	            moviercount[m*softmax+r]++;
	        }
	    }
	}
	
	public int train() {
	    
		DRand randn;
		
		/** Set initial weights */
	    int i, j, h, k;
	    
	    for (j=0; j<numItems; j++) {
	        for (i=0; i<totalFeatures; i++) {
	        	for(k=0;k<5;k++) {
	        		/** Normal Distribution */
	        		vishid[j][k][i] = 0.02 * randn.nextInt() - 0.01;	        	}
	        }
	    }

	    /** Set initial biases */
	    for(i=0;i<totalFeatures;i++) {
	        hidbiases[i] = 0.0;
	    }
	    
	    for (j=0; j<numItems; j++) {
	        int mtot = 0;
	        for(k=0;k<5;k++) {
	        	mtot += moviercount[j*softmax+k];
	        }
	        for (i=0; i<softmax; i++) {
	            visbiases[j][i] = Math.log(((double)moviercount[j*softmax+i])/((double)mtot));
	        }
	    }

	    /** Optimize current feature */
	    double nrmse = 2.0, last_rmse = 10.0;
	    double prmse = 0, last_prmse = 0;
	    double s;
	    int n;
	    int loopcount=0;
	    double EpsilonW  = this.epsilonw;
	    double EpsilonVB = this.epsilonvb;
	    double EpsilonHB = this.epsilonhb;
	    double Momentum  = this.momentum;
	    zero(CDinc, numItems, softmax, totalFeatures);
	    zero(visbiasinc, numItems, softmax);
	    zero(hidbiasinc, totalFeatures);
	    int tSteps = 1;

	    /** Iterate till improvement is less than e */ 
	    while (((nrmse < (last_rmse-e)) || loopcount < 14) && loopcount < 80  )  {

	        if ( loopcount >= 10 )
	            tSteps = 3+(loopcount-10)/5;

	        last_rmse = nrmse;
	        last_prmse = prmse;
	        loopcount++;
	        int ntrain = 0;
	        nrmse = 0.0;
	        s = 0.0;
	        n = 0;

	        if ( loopcount > 5 )
	            Momentum = finalMomentum;

	        zero(CDpos, numItems, softmax, totalFeatures);
	        zero(CDneg, numItems, softmax, totalFeatures);
	        zero(poshidact, totalFeatures);
	        zero(neghidact, totalFeatures);
	        zero(posvisact, numItems, softmax);
	        zero(negvisact, numItems, softmax);
	        zero(moviecount, numItems);

	        int u,m, f;
	        for(u=0;u<numUsers;u++) {

	            /** Probabilities */
	            zero(negvisprobs, numItems, softmax);
	            zero(nvp2, numItems, softmax);

	            /** Get data indices */
	            int base0=useridx[u][0]; //TODO: Replace
	            int d0=untrain(u); //TODO: Replace
	            int dall=unall(u); //TODO: Replace

	            /** For all rated movies, accumulate contributions to hidden units */
	            double[] sumW = new double[totalFeatures];
	            zero(sumW, totalFeatures);
	            for(j=0;j<d0;j++) {
	                int m=userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	                moviecount[m]++;

	                int r=(userent[base0+j]>>USER_LMOVIEMASK)&7; //TODO: Replace

	                /** Bias */
	                posvisact[m][r] += 1.0;
	 
	                /** For all hidden units */
	                for(h=0;h<totalFeatures;h++) {
	                    sumW[h]  += vishid[m][r][h];
	                }
	            }

	            /** Compute probabilities, and then sample the state of hidden units */
	            for(h=0;h<totalFeatures;h++) {
	                poshidprobs[h]  = 1.0/(1.0 + Math.exp(-sumW[h] - hidbiases[h]));
	                if  ( poshidprobs[h] >  randn.nextDouble() ) {
	                    poshidstates[h] = 1;
	                    poshidact[h] += 1.0;
	                } else {
	                    poshidstates[h] = 0;
	                }
	            }

	            /** Load up a copy of poshidstates for use in loop */
	            for ( h=0; h < totalFeatures; h++ ) 
	                curposhidstates[h] = poshidstates[h];

	            /** Make T steps of Contrastive Divergence */
	            int stepT = 0;
	            do {
	                /** Is the last pass through this loop? */
	                boolean finalTStep = (stepT+1 >= tSteps);
	                
	                int r;
	                int count = d0;
	                count += useridx[u][2];  /** For probe errors */
	                for(j=0;j<count;j++) {
	                    int m=userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	                    for(h=0;h<totalFeatures;h++) {
	                        /** Wherever sampled hidden states == 1, accumulate Weight values */
	                        if ( curposhidstates[h] == 1 ) {
	                            for(r=0;r<softmax;r++) {
	                                negvisprobs[m][r]  += vishid[m][r][h];
	                            }
	                        }

	                        /** Compute further accurate probabilities for RMSE reporting */
	                        if ( stepT == 0 ) {  
	                            for(r=0;r<softmax;r++) 
	                                nvp2[m][r] += poshidprobs[h] * vishid[m][r][h];
	                        }
	                    }

	                    for(i=0;i<5;i++) {
		                    negvisprobs[m][i]  = 1./(1 + Math.exp(-negvisprobs[m][i] - visbiases[m][i]));
	                    }

	                    /** Normalize probabilities */
	                    double tsum  = 0;
	                    for(i=0;i<5;i++) {
	                    	tsum += negvisprobs[m][i];
	                    }

	                    if ( tsum != 0 ) {
	                    	for(i=0;i<5;i++) {
		                        negvisprobs[m][i]  /= tsum;
	                    	}
	                    }
	                    
	                    /** Compute and Normalize more accurate RMSE reporting probabilities */
	                    if ( stepT == 0) {
	                    	for(i=0;i<5;i++) {
		                        nvp2[m][i]  = 1./(1 + Math.exp(-nvp2[m][i] - visbiases[m][i]));
	                    	}

	                        double tsum2 = 0;
		                    for(i=0;i<5;i++) {
		                    	tsum2 += nvp2[m][i];
		                    }
	                        if ( tsum2 != 0 ) {
			                    for(i=0;i<5;i++) {
			                    	tsum2 += nvp2[m][i];
			                    }
	                        }
	                    }

	                    double randval = randn.nextDouble();
	                    if ((randval -= negvisprobs[m][0]) <= 0.0)
	                        negvissoftmax[m] = 0;
	                    else if ((randval -= negvisprobs[m][1]) <= 0.0)
	                        negvissoftmax[m] = 1;
	                    else if ((randval -= negvisprobs[m][2]) <= 0.0)
	                        negvissoftmax[m] = 2;
	                    else if ((randval -= negvisprobs[m][3]) <= 0.0)
	                        negvissoftmax[m] = 3;
	                    else /** The case when ((randval -= negvisprobs[m][4]) <= 0.0) */
	                        negvissoftmax[m] = 4;

	                    /** If in training data, then train on it*/
	                    if ( j < d0 && finalTStep )  
	                        negvisact[m][negvissoftmax[m]] += 1.0;
	                }

	                /** For all rated movies, accumulate contributions to hidden units from sampled visible units */
	                zero(sumW, totalFeatures);
	                
	                for(j=0;j<d0;j++) {
	                    int m=userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	     
	                    /** For all hidden units */
	                    for(h=0;h<totalFeatures;h++) {
	                        sumW[h] += vishid[m][negvissoftmax[m]][h];
	                    }
	                }
                    /** For all hidden units */
	                for(h=0;h<totalFeatures;h++) {
	                    neghidprobs[h]  = 1./(1 + Math.exp(-sumW[h] - hidbiases[h]));

	                    /** Sample the hidden units state again. */
	                    if  ( neghidprobs[h] >  randn.nextDouble() ) {
	                        neghidstates[h]=1;
	                        if ( finalTStep )
	                            neghidact[h] += 1.0;
	                    } else {
	                        neghidstates[h]=0;
	                    }
	                }

	                /** Compute error rmse and prmse before we start iterating on T */
	                if ( stepT == 0 ) {

	                    /** Compute rmse on training data */
	                    for(j=0;j<d0;j++) {
	                        int m = userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	                        int r = (userent[base0+j]>>USER_LMOVIEMASK)&7; //TODO: Replace
	         
	                        double expectedV = nvp2[m][1] + 2.0 * nvp2[m][2] + 3.0 * nvp2[m][3] + 4.0 * nvp2[m][4];
	                        double vdelta = (((double)r)-expectedV);
	                        nrmse += (vdelta * vdelta);
	                    }
	                    
	                    ntrain+=d0;

	                    /** Sum up probe rmse */
	                    int base = useridx[u][0]; //TODO: Replace
	                    for(i=1;i<2;i++) base+=useridx[u][i]; //TODO: Replace
	                    int d=useridx[u][2]; //TODO: Replace
	                    for(i=0; i<d;i++) {
	                        int m=userent[base+i]&USER_MOVIEMASK; //TODO: Replace
	                        int r=(userent[base+i]>>USER_LMOVIEMASK)&7; //TODO: Replace

	                        double expectedV = nvp2[m][1] + 2.0 * nvp2[m][2] + 3.0 * nvp2[m][3] + 4.0 * nvp2[m][4];
	                        double vdelta = (((double)r)-expectedV);
	                        s+=vdelta*vdelta;
	                    }
	                    n+=d;
	                }

	                /** Are we looping again? Load curposvisstates */
	                if (!finalTStep) {
	                    for ( h=0; h < totalFeatures; h++ ) 
	                        curposhidstates[h] = neghidstates[h];
	                    zero(negvisprobs);
	                }
	            } while ( ++stepT < tSteps );

	            /** Accumulate contrastive divergence contributions */
	            for(j=0;j<d0;j++) {
	                int m=userent[base0+j]&USER_MOVIEMASK; //TODO: Replace
	                int r=(userent[base0+j]>>USER_LMOVIEMASK)&7;//TODO: Replace
	 
	                /** For all hidden units */
	                for(h=0;h<totalFeatures;h++) {
	                    if ( poshidstates[h] == 1 ) {
	                        CDpos[m][r][h] += 1.0;
	                    }
	                    CDneg[m][negvissoftmax[m]][h] += (double)neghidstates[h];
	                }
	            }

	            /** Update weights and biases */
	            int bSize = 100;
	            if(((u+1) % bSize)==0 || (u+1)==numUsers) {
	                int numcases = u % bSize;
	                numcases++;

	                /** Update weights */
	                for(m=0;m<numItems;m++) {
	                    if(moviecount[m] == 0) continue;

	                    /** For all hidden units */
	                    for(h=0;h<totalFeatures;h++) {
	                        /** For all softmax */
	                        int rr;
	                        for(rr=0;rr<softmax;rr++) {
	                            /** At the end compute average of CDpos and CDneg by dividing them by number of data points. */
	                            double CDp = CDpos[m][rr][h];
	                            double CDn = CDneg[m][rr][h];
	                            if ( CDp != 0.0 || CDn != 0.0 ) {
	                                CDp /= ((double)moviecount[m]);
	                                CDn /= ((double)moviecount[m]);

	                                /** Update weights and biases W = W + alpha*ContrastiveDivergence (biases are just weights to neurons that stay always 1.0) */
                  CDinc[m][rr][h] = Momentum * CDinc[m][rr][h] + EpsilonW
                      * ((CDp - CDn) - weightCost * vishid[m][rr][h]);
	                                vishid[m][rr][h] += CDinc[m][rr][h];
	                            } 
	                        }
	                    }

	                    /** Update visible softmax biases */
	                    int rr;
	                    for(rr=0;rr<softmax;rr++) {
	                        if(posvisact[m][rr] != 0.0 || negvisact[m][rr] != 0.0) {
	                            posvisact[m][rr] /= ((double)moviecount[m]);
	                            negvisact[m][rr] /= ((double)moviecount[m]);
	                            visbiasinc[m][rr] = Momentum * visbiasinc[m][rr] + EpsilonVB * ((posvisact[m][rr] - negvisact[m][rr]));
	                            visbiases[m][rr]  += visbiasinc[m][rr];
	                        }
	                    }
	                }

	                
	                /** Update hidden biases */
	                for(h=0;h<totalFeatures;h++) {
	                    if ( poshidact[h]  != 0.0 || neghidact[h]  != 0.0 ) {
	                        poshidact[h]  /= ((double)(numcases));
	                        neghidact[h]  /= ((double)(numcases));
	                        hidbiasinc[h] = Momentum * hidbiasinc[h] + EpsilonHB * ((poshidact[h] - neghidact[h]));
	                        hidbiases[h]  += hidbiasinc[h];
	                    }
	                }
	                
	    	        zero(CDpos, numItems, softmax, totalFeatures);
	    	        zero(CDneg, numItems, softmax, totalFeatures);
	    	        zero(poshidact, totalFeatures);
	    	        zero(neghidact, totalFeatures);
	    	        zero(posvisact, numItems, softmax);
	    	        zero(negvisact, numItems, softmax);
	    	        zero(moviecount, numItems);
	            }
	        }

	        nrmse = Math.sqrt(nrmse/ntrain);
	        prmse = Math.sqrt(s/n);
	        
	    /** Clip errors */
	    recordErrors();
	    
	    return 1;
	    }
	}

	private void recordErrors() {
		// TODO
		
	}

	private int unall(int u) {
		// TODO 
		return 0;
	}

	private int untrain(int u) {
		// TODO 
		return 0;
	}

	public float predictRating(int user, int item) {
		// TODO
    return 0;
  }
}
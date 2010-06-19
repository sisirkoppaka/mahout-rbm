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

import java.util.Collection;
import java.util.List;

import org.apache.mahout.cf.taste.impl.recommender.AbstractRecommender;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.impl.model.GenericDataModel;
import org.apache.mahout.cf.taste.impl.model.GenericPreference;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.cf.taste.common.NoSuchItemException;
import org.apache.mahout.cf.taste.common.NoSuchUserException;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A Recommender based on Restricted Boltzmann Machines.
 * Salakhutdinov R., Mnih A., Hinton G.E.(2007). Restricted Boltzmann Machines for Collaborative Filtering.
 */
public final class RBMRecommender extends AbstractRecommender {
  
  private static final Logger log = LoggerFactory.getLogger(RBMRecommender.class);
  
  private FastByIDMap<Integer> userMap;
  private FastByIDMap<Integer> itemMap;
  private RBM rbm;
  //private RBMState state;
  
  /** Default optimum constants for 100 hidden variables on the Netflix dataset. */
  private final int totalFeatures = 100;
  private final int softmax = 5;
  private final double epsilonw = 0.001; /** Learning rate for weights */
  private final double epsilonvb = 0.008; /** Learning rate for biases of visible units */
  private final double epsilonhb = 0.0006; /** Learning rate for biases of hidden units */
  private final double weightCost = 0.0001;
  private final double momentum = 0.8;
  private final double finalMomentum = 0.9;
  
  private final double e = 0.00002; /** Stop condition */
  
  public RBMRecommender(DataModel dataModel, int initialSteps) throws TasteException {
    super(dataModel);
        
    loadModelMaps(dataModel);
    
    rbm = new RBM(numUsers, numItems, numFeatures, defaultValue);
  
    rbm.train();
  }
  
  /**
   * @param numFeatures
   * @param initialSteps
   * @param totalFeatures
   * @param softmax
   * @param epsilonw
   * @param epsilonvb
   * @param epsilonhb
   * @param weightCost
   * @param momentum
   * @param finalMomentum
   */
  public RBMRecommender(DataModel dataModel, int initialSteps, int totalFeatures, int softmax, double epsilonw, double epsilonvb, double epsilonhb, double weightCost, double momentum, double finalMomentum) throws TasteException {
	  super(dataModel);
	  
	  this.totalFeatures = totalFeatures;
	  this.softmax = softmax;
	  this.epsilonw = epsilonw;
	  this.epsilonvb = epsilonvb;
	  this.epsilonhb = epsilonhb;
	  this.weightCost = weightCost;
	  this.momentum = momentum;
	  this.finalMomentum = finalMomentum;

	  loadModelMaps(dataModel);
	  
	  rbm = new RBM(numUsers, numItems, numFeatures, defaultValue);

	  rbm.train();
  }
  
  
  private void loadModelMaps(DataModel dataModel) throws TasteException {
	  int numUsers = dataModel.getNumUsers();
	  userMap = new FastByIDMap<Integer>(numUsers);
	    
	  int idx = 0;
	  LongPrimitiveIterator userIterator = dataModel.getUserIDs();
	  while (userIterator.hasNext()) {
	    userMap.put(userIterator.nextLong(), idx++);
	  }
	    
	  int numItems = dataModel.getNumItems();
	  itemMap = new FastByIDMap<Integer>(numItems);
	    
	  idx = 0;
	  LongPrimitiveIterator itemIterator = dataModel.getItemIDs();
	  while (itemIterator.hasNext()) {
	    itemMap.put(itemIterator.nextLong(), idx++);
	  }
	    
  }
  
  public void train(int steps) {
    rbm.train();
  }
  
  private float predictRating(int user, int item) {
    return (float) rbm.predictRating(user, item);
  }
  
  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    Integer useridx = userMap.get(userID);
    if (useridx == null) {
      throw new NoSuchUserException();
    }
    Integer itemidx = itemMap.get(itemID);
    if (itemidx == null) {
      throw new NoSuchItemException();
    }
    return predictRating(useridx, itemidx);
  }
  
  //TODO: Rework
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException {
    if (howMany < 1) {
      throw new IllegalArgumentException("1 is the minimum for howMany");
    }
    
    log.debug("Recommending items for user ID '{}'", userID);
    
    FastIDSet possibleItemIDs = getAllOtherItems(userID);
    
    TopItems.Estimator<Long> estimator = new Estimator(userID);
    
    List<RecommendedItem> topItems = TopItems.getTopItems(howMany,
        possibleItemIDs.iterator(), rescorer, estimator);
    
    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }
  
  /*
   * @Override public String toString() { return "RBMRecommender[totalFeatures:"
   * + totalFeatures + ']'; }
   */

  private final class Estimator implements TopItems.Estimator<Long> {
    
    private final long theUserID;
    
    private Estimator(long theUserID) {
      this.theUserID = theUserID;
    }
    
    @Override
    public double estimate(Long itemID) throws TasteException {
      return estimatePreference(theUserID, itemID);
    }
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    // TODO Auto-generated method stub
    
  }
  
}
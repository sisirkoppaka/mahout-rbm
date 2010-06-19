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

package org.apache.mahout.cf.taste.example.netflix;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.VLongWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

//This DataSetCreatorMapper and DataSetCreatorReducer may not be required. Remove stubs when done with rest of algo.

public class NetflixDatasetCreatorReducer extends MapReduceBase 
implements Reducer<VLongWritable,VLongWritable,VLongWritable,VectorWritable> {
  
  public void reducer(VLongWritable userID, Iterator<VLongWritable> itemPrefs,
      OutputCollector<VLongWritable, VectorWritable> output, Reporter reporter) throws IOException {
      
    Vector userVector = RandomAccessSparseVector(Integer.MAX_VALUE, 100);
    while(itemPrefs.hasNext()) {
      VLongWritable itemPref = itemPrefs.next();
      userVector.set(itemPref.get(), 1.0f);
    }
    
    output.collect(userID, new VectorWritable(userVector));
  }
}

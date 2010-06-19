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

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.hadoop.DistributedRowMatrix;

public class RBMInputReducer extends Reducer<IntWritable, DistributedRowMatrix.MatrixEntryWritable, 
        IntWritable, VectorWritable> {
  
  @Override
  protected void reduce(IntWritable record, 
              Iterable<DistributedRowMatrix.MatrixEntryWritable> recordEntries,
              Context context) 
              throws IOException, InterruptedException {
    RandomAccessSparseVector toWrite = new RandomAccessSparseVector(Integer.MAX_VALUE, 100); //100? or something else?

    for (DistributedRowMatrix.MatrixEntryWritable entryItem : recordEntries) {
      toWrite.setQuick(entryItem.getCol(), entryItem.getVal());
    }
    SequentialAccessSparseVector output = new SequentialAccessSparseVector(toWrite);
    context.write(record, new VectorWritable(output));
  }
}

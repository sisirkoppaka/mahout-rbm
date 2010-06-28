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

import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Varint;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Iterator;

public class RBMStateWritable extends Configured implements Writable {

  private RBMState state;

  public RBMStateWritable() {
  }

  public RBMStateWritable(RBMState state) {
    this.state = state;
  }

  /**
   * @return {@link RBMState} that this {@link RBMStateWritable} is to write, or has
   *  just read
   */
  public RBMState get() {
    return state;
  }

  public void set(RBMState state) {
    this.state = state;
  }

  @Override
  public void write(DataOutput out) throws IOException {

    int i,j,k;
    Varint.writeUnsignedVarInt(6, out);
    Varint.writeUnsignedVarInt(3, out);
    Varint.writeUnsignedVarInt(state.numItems, out);
    Varint.writeUnsignedVarInt(state.softmax, out);
    Varint.writeUnsignedVarInt(state.totalFeatures, out);
    for(i=0;i<state.numItems;i++) {
      for(j=0;j<state.softmax;j++) {
        for(k=0;k<state.totalFeatures;k++) {
          out.writeDouble(state.vishid[i][j][k]);
        }
      }
    }
    Varint.writeUnsignedVarInt(2, out);
    Varint.writeUnsignedVarInt(state.numItems, out);
    Varint.writeUnsignedVarInt(state.softmax, out);
    for(i=0;i<state.numItems;i++) {
      for(j=0;j<state.softmax;j++) {
        out.writeDouble(state.visbiases[i][j]);
      }
    }
    Varint.writeUnsignedVarInt(1, out);
    Varint.writeUnsignedVarInt(state.totalFeatures, out);
    for(i=0;i<state.totalFeatures;i++) {
      out.writeDouble(state.hidbiases[state.totalFeatures]);
    }
    Varint.writeUnsignedVarInt(3, out);
    Varint.writeUnsignedVarInt(state.numItems, out);
    Varint.writeUnsignedVarInt(state.softmax, out);
    Varint.writeUnsignedVarInt(state.totalFeatures, out);
    for(i=0;i<state.numItems;i++) {
      for(j=0;j<state.softmax;j++) {
        for(k=0;k<state.totalFeatures;k++) {
          out.writeDouble(state.CDinc[i][j][k]);
        }
      }
    }
    Varint.writeUnsignedVarInt(1, out);
    Varint.writeUnsignedVarInt(state.totalFeatures, out);
    for(i=0;i<state.totalFeatures;i++) {
      out.writeDouble(state.hidbiasinc[state.totalFeatures]);
    }
    Varint.writeUnsignedVarInt(2, out);
    Varint.writeUnsignedVarInt(state.numItems, out);
    Varint.writeUnsignedVarInt(state.softmax, out);
    for(i=0;i<state.numItems;i++) {
      for(j=0;j<state.softmax;j++) {
        out.writeDouble(state.visbiasinc[i][j]);
      }
    }
  }

  @Override
  public void readFields(DataInput in) throws IOException {

    int size = Varint.readUnsignedVarInt(in);
    RBMState s = new RBMState();
    
    int i,j,k;
    int a,b,c;
    Varint.readUnsignedVarInt(in);
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    b = Varint.readUnsignedVarInt(in);
    c = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      for(j=0;j<b;j++) {
        for(k=0;k<c;k++) {
          s.vishid[i][j][k]=in.readDouble();
        }
      }
    }
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    b = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      for(j=0;j<b;j++) {
        s.visbiases[i][j]=in.readDouble();
      }
    }
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      s.hidbiases[i]=in.readDouble();
    }
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    b = Varint.readUnsignedVarInt(in);
    c = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      for(j=0;j<b;j++) {
        for(k=0;k<c;k++) {
          s.CDinc[i][j][k]=in.readDouble();
        }
      }
    }
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      s.hidbiasinc[i]=in.readDouble();
    }
    Varint.readUnsignedVarInt(in);
    a = Varint.readUnsignedVarInt(in);
    b = Varint.readUnsignedVarInt(in);
    for(i=0;i<a;i++) {
      for(j=0;j<b;j++) {
        s.visbiasinc[i][j]=in.readDouble();
      }
    }
    state = s;
  }

  /** Write the RBMState to the output */
  public static void writeRBMState(DataOutput out, RBMState state) throws IOException {
    new RBMStateWritable(state).write(out);
  }

}
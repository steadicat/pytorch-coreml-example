//
//  SplitBridge.m
//  Movement
//
//  Created by Stefano J. Attardi on 1/19/18.
//  Copyright © 2018 Rational Creation. All rights reserved.
//

#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(Split, NSObject)

RCT_EXTERN_METHOD(split:(NSArray<NSArray<NSNumber *> *> *)points callback:(RCTResponseSenderBlock *)callback)

@end

#import <React/RCTConvert.h>

@interface RCTConvert (RCTConvertNSNumberArrayArray)
@end

@implementation RCTConvert (RCTConvertNSNumberArrayArray)
+ (NSArray<NSArray<NSNumber *> *> *)NSNumberArrayArray:(id)json
{
  return RCTConvertArrayValue(@selector(NSNumberArray:), json);
}
@end


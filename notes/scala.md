```scala

case BooleanType => types.ScalarType.Boolean
      case ByteType => types.ScalarType.Byte
      case ShortType => types.ScalarType.Short
      case IntegerType => types.ScalarType.Int
      case LongType => types.ScalarType.Long
      case FloatType => types.ScalarType.Float
      case DoubleType => types.ScalarType.Double
      case _: DecimalType => types.ScalarType.Double
      case StringType => types.ScalarType.String.setNullable(field.nullable)
      case ArrayType(ByteType, _) => types.ListType.Byte
      case ArrayType(BooleanType, _) => types.ListType.Boolean
      case ArrayType(ShortType, _) => types.ListType.Short
      case ArrayType(IntegerType, _) => types.ListType.Int
      case ArrayType(LongType, _) => types.ListType.Long
      case ArrayType(FloatType, _) => types.ListType.Float
      case ArrayType(DoubleType, _) => types.ListType.Double
case ArrayType(StringType, _) => types.ListType.String
```

* http://mleap-docs.combust.ml/mleap-runtime/storing.html
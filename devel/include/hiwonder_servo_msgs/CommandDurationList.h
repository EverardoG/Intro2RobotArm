// Generated by gencpp from file hiwonder_servo_msgs/CommandDurationList.msg
// DO NOT EDIT!


#ifndef HIWONDER_SERVO_MSGS_MESSAGE_COMMANDDURATIONLIST_H
#define HIWONDER_SERVO_MSGS_MESSAGE_COMMANDDURATIONLIST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace hiwonder_servo_msgs
{
template <class ContainerAllocator>
struct CommandDurationList_
{
  typedef CommandDurationList_<ContainerAllocator> Type;

  CommandDurationList_()
    : duration(0.0)
    , ids()
    , positions()  {
    }
  CommandDurationList_(const ContainerAllocator& _alloc)
    : duration(0.0)
    , ids(_alloc)
    , positions(_alloc)  {
  (void)_alloc;
    }



   typedef double _duration_type;
  _duration_type duration;

   typedef std::vector<uint16_t, typename ContainerAllocator::template rebind<uint16_t>::other >  _ids_type;
  _ids_type ids;

   typedef std::vector<double, typename ContainerAllocator::template rebind<double>::other >  _positions_type;
  _positions_type positions;





  typedef boost::shared_ptr< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> const> ConstPtr;

}; // struct CommandDurationList_

typedef ::hiwonder_servo_msgs::CommandDurationList_<std::allocator<void> > CommandDurationList;

typedef boost::shared_ptr< ::hiwonder_servo_msgs::CommandDurationList > CommandDurationListPtr;
typedef boost::shared_ptr< ::hiwonder_servo_msgs::CommandDurationList const> CommandDurationListConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator1> & lhs, const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator2> & rhs)
{
  return lhs.duration == rhs.duration &&
    lhs.ids == rhs.ids &&
    lhs.positions == rhs.positions;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator1> & lhs, const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace hiwonder_servo_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ee846be6e4a1d12d4044e7694b9b051b";
  }

  static const char* value(const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xee846be6e4a1d12dULL;
  static const uint64_t static_value2 = 0x4044e7694b9b051bULL;
};

template<class ContainerAllocator>
struct DataType< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "hiwonder_servo_msgs/CommandDurationList";
  }

  static const char* value(const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float64 duration\n"
"uint16[] ids\n"
"float64[] positions\n"
;
  }

  static const char* value(const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.duration);
      stream.next(m.ids);
      stream.next(m.positions);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct CommandDurationList_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::hiwonder_servo_msgs::CommandDurationList_<ContainerAllocator>& v)
  {
    s << indent << "duration: ";
    Printer<double>::stream(s, indent + "  ", v.duration);
    s << indent << "ids[]" << std::endl;
    for (size_t i = 0; i < v.ids.size(); ++i)
    {
      s << indent << "  ids[" << i << "]: ";
      Printer<uint16_t>::stream(s, indent + "  ", v.ids[i]);
    }
    s << indent << "positions[]" << std::endl;
    for (size_t i = 0; i < v.positions.size(); ++i)
    {
      s << indent << "  positions[" << i << "]: ";
      Printer<double>::stream(s, indent + "  ", v.positions[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // HIWONDER_SERVO_MSGS_MESSAGE_COMMANDDURATIONLIST_H

/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 2.0.9
 *
 * Do not make changes to this file unless you know what you are doing--modify
 * the SWIG interface file instead.
 * ----------------------------------------------------------------------------- */

namespace org.doubango.ultimateAlpr.Sdk {

using System;
using System.Runtime.InteropServices;

public class UltAlprSdkResult : IDisposable {
  private HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal UltAlprSdkResult(IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new HandleRef(this, cPtr);
  }

  internal static HandleRef getCPtr(UltAlprSdkResult obj) {
    return (obj == null) ? new HandleRef(null, IntPtr.Zero) : obj.swigCPtr;
  }

  ~UltAlprSdkResult() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          ultimateAlprSdkPINVOKE.delete_UltAlprSdkResult(swigCPtr);
        }
        swigCPtr = new HandleRef(null, IntPtr.Zero);
      }
      GC.SuppressFinalize(this);
    }
  }

  public UltAlprSdkResult(int code, string phrase, string json, uint numPlates) : this(ultimateAlprSdkPINVOKE.new_UltAlprSdkResult__SWIG_0(code, phrase, json, numPlates), true) {
  }

  public UltAlprSdkResult(int code, string phrase, string json) : this(ultimateAlprSdkPINVOKE.new_UltAlprSdkResult__SWIG_1(code, phrase, json), true) {
  }

  public int code() {
    int ret = ultimateAlprSdkPINVOKE.UltAlprSdkResult_code(swigCPtr);
    return ret;
  }

  public string phrase() {
    string ret = ultimateAlprSdkPINVOKE.UltAlprSdkResult_phrase(swigCPtr);
    return ret;
  }

  public string json() {
    string ret = ultimateAlprSdkPINVOKE.UltAlprSdkResult_json(swigCPtr);
    return ret;
  }

  public uint numPlates() {
    uint ret = ultimateAlprSdkPINVOKE.UltAlprSdkResult_numPlates(swigCPtr);
    return ret;
  }

  public bool isOK() {
    bool ret = ultimateAlprSdkPINVOKE.UltAlprSdkResult_isOK(swigCPtr);
    return ret;
  }

}

}

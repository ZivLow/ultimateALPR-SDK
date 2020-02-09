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

class ultimateAlprSdkPINVOKE {

  protected class SWIGExceptionHelper {

    public delegate void ExceptionDelegate(string message);
    public delegate void ExceptionArgumentDelegate(string message, string paramName);

    static ExceptionDelegate applicationDelegate = new ExceptionDelegate(SetPendingApplicationException);
    static ExceptionDelegate arithmeticDelegate = new ExceptionDelegate(SetPendingArithmeticException);
    static ExceptionDelegate divideByZeroDelegate = new ExceptionDelegate(SetPendingDivideByZeroException);
    static ExceptionDelegate indexOutOfRangeDelegate = new ExceptionDelegate(SetPendingIndexOutOfRangeException);
    static ExceptionDelegate invalidCastDelegate = new ExceptionDelegate(SetPendingInvalidCastException);
    static ExceptionDelegate invalidOperationDelegate = new ExceptionDelegate(SetPendingInvalidOperationException);
    static ExceptionDelegate ioDelegate = new ExceptionDelegate(SetPendingIOException);
    static ExceptionDelegate nullReferenceDelegate = new ExceptionDelegate(SetPendingNullReferenceException);
    static ExceptionDelegate outOfMemoryDelegate = new ExceptionDelegate(SetPendingOutOfMemoryException);
    static ExceptionDelegate overflowDelegate = new ExceptionDelegate(SetPendingOverflowException);
    static ExceptionDelegate systemDelegate = new ExceptionDelegate(SetPendingSystemException);

    static ExceptionArgumentDelegate argumentDelegate = new ExceptionArgumentDelegate(SetPendingArgumentException);
    static ExceptionArgumentDelegate argumentNullDelegate = new ExceptionArgumentDelegate(SetPendingArgumentNullException);
    static ExceptionArgumentDelegate argumentOutOfRangeDelegate = new ExceptionArgumentDelegate(SetPendingArgumentOutOfRangeException);

    [DllImport("ultimateAlprSdk", EntryPoint="SWIGRegisterExceptionCallbacks_ultimateAlprSdk")]
    public static extern void SWIGRegisterExceptionCallbacks_ultimateAlprSdk(
                                ExceptionDelegate applicationDelegate,
                                ExceptionDelegate arithmeticDelegate,
                                ExceptionDelegate divideByZeroDelegate, 
                                ExceptionDelegate indexOutOfRangeDelegate, 
                                ExceptionDelegate invalidCastDelegate,
                                ExceptionDelegate invalidOperationDelegate,
                                ExceptionDelegate ioDelegate,
                                ExceptionDelegate nullReferenceDelegate,
                                ExceptionDelegate outOfMemoryDelegate, 
                                ExceptionDelegate overflowDelegate, 
                                ExceptionDelegate systemExceptionDelegate);

    [DllImport("ultimateAlprSdk", EntryPoint="SWIGRegisterExceptionArgumentCallbacks_ultimateAlprSdk")]
    public static extern void SWIGRegisterExceptionCallbacksArgument_ultimateAlprSdk(
                                ExceptionArgumentDelegate argumentDelegate,
                                ExceptionArgumentDelegate argumentNullDelegate,
                                ExceptionArgumentDelegate argumentOutOfRangeDelegate);

    static void SetPendingApplicationException(string message) {
      SWIGPendingException.Set(new System.ApplicationException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingArithmeticException(string message) {
      SWIGPendingException.Set(new System.ArithmeticException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingDivideByZeroException(string message) {
      SWIGPendingException.Set(new System.DivideByZeroException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingIndexOutOfRangeException(string message) {
      SWIGPendingException.Set(new System.IndexOutOfRangeException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingInvalidCastException(string message) {
      SWIGPendingException.Set(new System.InvalidCastException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingInvalidOperationException(string message) {
      SWIGPendingException.Set(new System.InvalidOperationException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingIOException(string message) {
      SWIGPendingException.Set(new System.IO.IOException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingNullReferenceException(string message) {
      SWIGPendingException.Set(new System.NullReferenceException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingOutOfMemoryException(string message) {
      SWIGPendingException.Set(new System.OutOfMemoryException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingOverflowException(string message) {
      SWIGPendingException.Set(new System.OverflowException(message, SWIGPendingException.Retrieve()));
    }
    static void SetPendingSystemException(string message) {
      SWIGPendingException.Set(new System.SystemException(message, SWIGPendingException.Retrieve()));
    }

    static void SetPendingArgumentException(string message, string paramName) {
      SWIGPendingException.Set(new System.ArgumentException(message, paramName, SWIGPendingException.Retrieve()));
    }
    static void SetPendingArgumentNullException(string message, string paramName) {
      Exception e = SWIGPendingException.Retrieve();
      if (e != null) message = message + " Inner Exception: " + e.Message;
      SWIGPendingException.Set(new System.ArgumentNullException(paramName, message));
    }
    static void SetPendingArgumentOutOfRangeException(string message, string paramName) {
      Exception e = SWIGPendingException.Retrieve();
      if (e != null) message = message + " Inner Exception: " + e.Message;
      SWIGPendingException.Set(new System.ArgumentOutOfRangeException(paramName, message));
    }

    static SWIGExceptionHelper() {
      SWIGRegisterExceptionCallbacks_ultimateAlprSdk(
                                applicationDelegate,
                                arithmeticDelegate,
                                divideByZeroDelegate,
                                indexOutOfRangeDelegate,
                                invalidCastDelegate,
                                invalidOperationDelegate,
                                ioDelegate,
                                nullReferenceDelegate,
                                outOfMemoryDelegate,
                                overflowDelegate,
                                systemDelegate);

      SWIGRegisterExceptionCallbacksArgument_ultimateAlprSdk(
                                argumentDelegate,
                                argumentNullDelegate,
                                argumentOutOfRangeDelegate);
    }
  }

  protected static SWIGExceptionHelper swigExceptionHelper = new SWIGExceptionHelper();

  public class SWIGPendingException {
    [ThreadStatic]
    private static Exception pendingException = null;
    private static int numExceptionsPending = 0;

    public static bool Pending {
      get {
        bool pending = false;
        if (numExceptionsPending > 0)
          if (pendingException != null)
            pending = true;
        return pending;
      } 
    }

    public static void Set(Exception e) {
      if (pendingException != null)
        throw new ApplicationException("FATAL: An earlier pending exception from unmanaged code was missed and thus not thrown (" + pendingException.ToString() + ")", e);
      pendingException = e;
      lock(typeof(ultimateAlprSdkPINVOKE)) {
        numExceptionsPending++;
      }
    }

    public static Exception Retrieve() {
      Exception e = null;
      if (numExceptionsPending > 0) {
        if (pendingException != null) {
          e = pendingException;
          pendingException = null;
          lock(typeof(ultimateAlprSdkPINVOKE)) {
            numExceptionsPending--;
          }
        }
      }
      return e;
    }
  }


  protected class SWIGStringHelper {

    public delegate string SWIGStringDelegate(string message);
    static SWIGStringDelegate stringDelegate = new SWIGStringDelegate(CreateString);

    [DllImport("ultimateAlprSdk", EntryPoint="SWIGRegisterStringCallback_ultimateAlprSdk")]
    public static extern void SWIGRegisterStringCallback_ultimateAlprSdk(SWIGStringDelegate stringDelegate);

    static string CreateString(string cString) {
      return cString;
    }

    static SWIGStringHelper() {
      SWIGRegisterStringCallback_ultimateAlprSdk(stringDelegate);
    }
  }

  static protected SWIGStringHelper swigStringHelper = new SWIGStringHelper();


  static ultimateAlprSdkPINVOKE() {
  }


  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_ULTALPR_SDK_VERSION_MAJOR_get")]
  public static extern int ULTALPR_SDK_VERSION_MAJOR_get();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_ULTALPR_SDK_VERSION_MINOR_get")]
  public static extern int ULTALPR_SDK_VERSION_MINOR_get();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_ULTALPR_SDK_VERSION_MICRO_get")]
  public static extern int ULTALPR_SDK_VERSION_MICRO_get();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_new_UltAlprSdkResult__SWIG_0")]
  public static extern IntPtr new_UltAlprSdkResult__SWIG_0(int jarg1, string jarg2, string jarg3, uint jarg4);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_new_UltAlprSdkResult__SWIG_1")]
  public static extern IntPtr new_UltAlprSdkResult__SWIG_1(int jarg1, string jarg2, string jarg3);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_delete_UltAlprSdkResult")]
  public static extern void delete_UltAlprSdkResult(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkResult_code")]
  public static extern int UltAlprSdkResult_code(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkResult_phrase")]
  public static extern string UltAlprSdkResult_phrase(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkResult_json")]
  public static extern string UltAlprSdkResult_json(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkResult_numPlates")]
  public static extern uint UltAlprSdkResult_numPlates(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkResult_isOK")]
  public static extern bool UltAlprSdkResult_isOK(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_new_UltAlprSdkParallelDeliveryCallback")]
  public static extern IntPtr new_UltAlprSdkParallelDeliveryCallback();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_delete_UltAlprSdkParallelDeliveryCallback")]
  public static extern void delete_UltAlprSdkParallelDeliveryCallback(HandleRef jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkParallelDeliveryCallback_onNewResult")]
  public static extern void UltAlprSdkParallelDeliveryCallback_onNewResult(HandleRef jarg1, HandleRef jarg2);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkParallelDeliveryCallback_director_connect")]
  public static extern void UltAlprSdkParallelDeliveryCallback_director_connect(HandleRef jarg1, UltAlprSdkParallelDeliveryCallback.SwigDelegateUltAlprSdkParallelDeliveryCallback_0 delegate0);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_init__SWIG_0")]
  public static extern IntPtr UltAlprSdkEngine_init__SWIG_0(string jarg1, HandleRef jarg2);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_init__SWIG_1")]
  public static extern IntPtr UltAlprSdkEngine_init__SWIG_1(string jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_init__SWIG_2")]
  public static extern IntPtr UltAlprSdkEngine_init__SWIG_2();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_deInit")]
  public static extern IntPtr UltAlprSdkEngine_deInit();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_0")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_0(int jarg1, IntPtr jarg2, uint jarg3, uint jarg4, uint jarg5, int jarg6);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_1")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_1(int jarg1, IntPtr jarg2, uint jarg3, uint jarg4, uint jarg5);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_2")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_2(int jarg1, IntPtr jarg2, uint jarg3, uint jarg4);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_3")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_3(int jarg1, IntPtr jarg2, IntPtr jarg3, IntPtr jarg4, uint jarg5, uint jarg6, uint jarg7, uint jarg8, uint jarg9, uint jarg10, int jarg11);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_4")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_4(int jarg1, IntPtr jarg2, IntPtr jarg3, IntPtr jarg4, uint jarg5, uint jarg6, uint jarg7, uint jarg8, uint jarg9, uint jarg10);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_process__SWIG_5")]
  public static extern IntPtr UltAlprSdkEngine_process__SWIG_5(int jarg1, IntPtr jarg2, IntPtr jarg3, IntPtr jarg4, uint jarg5, uint jarg6, uint jarg7, uint jarg8, uint jarg9);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_requestRuntimeLicenseKey__SWIG_0")]
  public static extern IntPtr UltAlprSdkEngine_requestRuntimeLicenseKey__SWIG_0(bool jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_requestRuntimeLicenseKey__SWIG_1")]
  public static extern IntPtr UltAlprSdkEngine_requestRuntimeLicenseKey__SWIG_1();

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_UltAlprSdkEngine_warmUp")]
  public static extern IntPtr UltAlprSdkEngine_warmUp(int jarg1);

  [DllImport("ultimateAlprSdk", EntryPoint="CSharp_delete_UltAlprSdkEngine")]
  public static extern void delete_UltAlprSdkEngine(HandleRef jarg1);
}

}

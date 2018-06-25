# letcode 简单等级刷题
### 1.两数之和
>给定一个整数数列，找出其中和为特定值的那两个数。
>你可以假设每个输入都只会有一种答案，同样的元素不能被重用。

**示例：**
>给定 nums = [2, 7, 11, 15], target = 9
因为 nums[0] + nums[1] = 2 + 7 = 9
所以返回 [0, 1]

**我 6480ms**
```python
   for i in range(len(nums)-1):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    return [i,j]
```
**参考 48ms**
```python
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        dic = dict()
        for index,value in enumerate(nums):
            sub = target - value
            if sub in dic:#只对字典的键进行查找
                return [dic[sub],index]
            else:
                dic[value] = index#将nums数组值和索引值存储在字典中，其中value为键，索引为值
```
### 7.颠倒整数
>给定一个范围为 32 位 int 的整数，将其颠倒。

例 1:

输入: 123
输出:  321
 

例 2:

输入: -123
输出: -321
 

例 3:

输入: 120
输出: 21
 

注意:

假设我们的环境只能处理 32 位 int 范围内的整数。根据这个假设，如果颠倒后的结果超过这个范围，则返回 0。

我：
```python
 l = list(str(x))
        n = len(l)
        for i in range(n-1,0,-1):
            if l[i] != '0':
                l = l[:i+1]
                break
        if l[0] =='-':
                    l = l[:0:-1]
                    l.insert(0,'-') 
        else:
            l = l[::-1]
        return int(''.join(l))
```
other1:
```python
 x = int(str(x)[::-1]) if x >0 else -int(str(-x)[::-1])
return x if x < 2147483648 and x >-2147483648 else 0
```
other2:
```python
        sign = x < 0 and -1 or 1
        x = abs(x)
        ans = 0
        while x:
            ans = ans * 10 + x % 10
            x /= 10
        return sign * ans if ans <= 0x7fffffff else 0
```

>总结：
1.python中字符串就可以完成分片操作，不需要再转换为list。用一个新的数据对象时要考虑清楚为什么用；
2.正数和负数只相差一个符号！如例2中设置一个符号位。1与任何数与都为任何数，0与任何数与都为0。与1 或是为了将0变为1.
3.对于32位的限制，去掉一个符号位，还剩31位。所以范围是正负2的31次方之间，可以像例2一样表示0x7ffffff.

### 9.回文数
判断一个整数是否是回文数。不能使用辅助空间。
我(这里要表扬一下，跟参考的写得差不多)
```python
        m = x
        s = 0
        while m!=0:
            s = s*10+(m%10)
            m = m//10
        return s == x#参考里用这一句替代了我以下的if语句
        ```
        if s == x:
            return True
        else:
            return False
        ```
```
缺点就是速度慢。
### 13.罗马数字转整数
>给定一个罗马数字，将其转换成整数。

返回的结果要求在 1 到 3999 的范围内。

我
```python
        d = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        result = 0
        i = 0
        while i < len(s)-1:
            if d[s[i]] >= d[s[i+1]]:
                result+= d[s[i]]
                i+=1
            else:
                temp = d[s[i+1]]-d[s[i]]
                result+=temp
                i+=2
        if i == len(s)-1:
            result+=d[s[i]]
        return result if 1 <= result <= 3999 else False    
```
别人
```python
        d = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
        result = 0
        for i in range(len(s)-1):
            if d[s[i]] < d[s[i+1]]:
                result-= d[s[i]]
            else:
                result+=d[s[i]]
        result+=d[s[len(s)-1]]
        return result if 1 < result < 3999 else False
```
相关知识点在word文档中。
总结：思想要灵活，只知道加，不知道换个角度思考，不能加我可以减呐！
### 14.最长公共前缀
>编写一个函数来查找字符串数组中最长的公共前缀字符串。

解题分析：最长公共前缀指的是字符串数组中所有公共最长的前缀。
如果是空串的话，那么说明前缀就是“”
如果都是以“ ”开头的，那么就是“ ”
然后最长的前缀不会超过最短的字符串，那么可以遍历最短的字符串的长度，依次比较。
第一步：找出长度最短的字符串；
第二步：依次与长度最短的字符串比较。
我
```python
if strs == []:
            return ''
        common = []
        flag = 0
        def minimum(s):
            m = list(map(len,s))
            return s[m.index(min(m))]
        temp = minimum(strs)
        if temp == '':
            return ''
        a = temp[0]
        for i in strs:
            if i[0] != a:
                return ''
        for i in range(len(temp)):
            for j in strs:
                if temp[i]!=j[i]:
                    flag = 1
                    break
            if flag == 0:
                common.append(temp[i])
        return ''.join(common)
```
二刷
```python
        if not strs:
            return ''
        a = list(map(len,strs))
        temp = strs[a.index(min(a))]
        r = ''
        for i in range(min(a)):
            for j in strs:
                if j[i] != temp[i]:
                    return r
            r+=temp[i]
        return r
```

别人
```python
        if not strs:
            return ""
        if len(strs) == 1:
            return strs[0]
        minl = min([len(x) for x in strs])
        end = 0
        while end < minl:
            for i in range(1,len(strs)):
                if strs[i][end]!= strs[i-1][end]:
                    return strs[0][:end]
            end += 1
        return strs[0][:end]
```
大神
```python
        res = ""
        if len(strs) == 0:
            return ""
        for each in zip(*strs):#zip()函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
            if len(set(each)) == 1:#利用集合创建一个无序不重复元素集
                res += each[0]
            else:
                return res
        return res
```
总结：1.不需要求出最短字符串是哪个，只需要求出最短字符串的长度即可。
2.字符串组的几种情况的判断应更清晰明了。
1）字符串组为空[]
2）公共字符串为空['','']
3)正常情况
### 20.有效的括号
>给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
括号必须以正确的顺序关闭，"()" 和 "()[]{}" 是有效的但是 "(]" 和 "([)]" 不是。

我(我的特点是写代码之前有个大致的思路就开始写，然后边写边改，导致无数个if，else逻辑不清。)
```python
        d = {'(':')','{':'}','[':']'}
        l = ''
        if len(s) == 1 or len(s) == 0:
            return False
        for  i in range(len(s)):
            print(s[i])
            if s[i] in d:
                l+=d[s[i]]
                print(l)
                if i ==len(s)-1:
                    return False
            elif s[i] == l[len(l)-1:]:
                l = l[:len(l)-1]
                print(l)
                if i == len(s)-1:
                    if l == '':
                        return True  
                    else:
                        return False
            else:
                return False   
```
别人一目了然
```python
        if len(s) % 2 == 1 or len(s) == 0:#这一步真的机智，我只会想到输入一个的状况。。。
            return False
        
        d = {'{': '}', '[': ']', '(': ')'}
        stack = []
        for i in s:
            # in stack
            if i in d:
                stack.append(i)
            else:
                if not stack or d[stack.pop()] != i:
                    return False
        ```
        else:
            if stack:
                return False
            
        return True
        ```
        return stack ==[]#以上四条语句可以用这条语句替代
```
最强王者
```python
        a = {')':'(', ']':'[', '}':'{'}
        l = [None]#设置None是为了排除空值的情况！
        for i in s:
            if i in a and a[i] == l[-1]:
                l.pop()
            else:
                l.append(i)
        return len(l)==1#用来排除空值的情况
```
总结：
1.拿到一个题首先要考虑有几种情况：
1)空和奇数个直接排除
2)剩下偶数个，若第一个就是右括号直接排除，不是右括号再入栈。
2.画个流程图辅助。
3.for else的用法。for循环完后执行else语句块，若for中有break，则跳过else。
4.l[-1]为l的最后一个元素。
### 21.合并两个有序链表
>合并两个已排序的链表，并将其作为一个新列表返回。新列表应该通过拼接前两个列表的节点来完成。 

示例
>输入：1->2->4, 1->3->4
输出：1->1->2->3->4->4

这题我写不出来，参考别人的写了好几遍依旧不流畅。所以这题应经常刷。
```python
 dummy = ListNode(0)
        s = dummy
        while l1 and l2:
            if l1.val > l2.val:
                s.next = l2
                l2 = l2.next
            else:
                s.next = l1
                l1 = l1.next
        if l1:
            while l1:
                s.next = l1
        if l2:
            while l2:
                s.next = l2
        return dummy.next
```
总结：这题关于链表的操作不熟练，从中看出我对类的理解还不透彻，对数据结构的理解也不透彻。
###26 从有序数组中删除重复项
>描述：
给定一个有序数组，你需要原地删除其中的重复内容，使每个元素只出现一次,并返回新的长度。
不要另外定义一个数组，您必须通过用 O(1) 额外内存原地修改输入的数组来做到这一点。

示例
>给定数组: nums = [1,1,2],
你的函数应该返回新长度 2, 并且原数组nums的前两个元素必须是1和2
不需要理会新的数组长度后面的元素

我,这是我第一个提交一次就通过的代码。
```python
        i = 0
        l = len(nums)
        if l:
            while i < l-1:
                if nums[i] == nums[i+1]:
                    nums.pop(i+1)
                    l-=1
                    j = i+1
                    while j < l:
                        if nums[i] != nums[j]:
                            break
                        else:
                            nums.pop(j)
                            l-=1
                i+=1
        return len(nums)
```
开心得不想看别人的代码了。
### 27.移除元素
>给定一个数组和一个值，在这个数组中原地移除指定值和返回移除后新的数组长度。
不要为其他数组分配额外空间，你必须使用 O(1) 的额外内存原地修改这个输入数组。
元素的顺序可以改变。超过返回的新的数组长度以外的数据无论是什么都没关系。

示例
>给定 nums = [3,2,2,3]，val = 3，
你的函数应该返回 长度 = 2，数组的前两个元素是 2。

我:不得了，这题十分钟就做好，零bug！！超越昨天一次性通过所有测试！！！
```python
        l = len(nums)
        if l == 0:
            return 0
        i = 0
        while i < l:
            if nums[i] == val:
                nums.pop(i)
                l-=1
            else:
                i+=1
        return len(nums)
```
总结：这一题和上一题都使用了，list.pop()方法，数组长度一直在发生变化。这个时候想要遍历数组的话，使用while比较合适，用for循环的话，遍历的索引值不方便控制其根据list长度变化而跟着变化。
比如：
```python
l = [1,2,3,4]
for i in l:
    print(i)
    l.pop(1)
    print(l)
```
以上代码块的运行结果为：
1#此时i访问的是l的第一个元素
2
[1, 3, 4]
3#此时i访问的是l的第二个元素
3
[1, 4]#下一步i应该访问l的第三个元素，但因为此时l只剩余两个元素，故运行结束

### 28实现strStr()
>实现 strStr()。
返回蕴含在 haystack 中的 needle 的第一个字符的索引，如果 needle 不是 haystack 的一部分则返回 -1 。

示例
>输入: haystack = "hello", needle = "ll"
输出: 2

>输入: haystack = "aaaaa", needle = "bba"
输出: -1

我：一次过。第一次感受到占了Python刷题的大便宜了，但是感觉不妥，还是应该正正经经从逻辑上写代码，不过三行代码是真的爽哈哈哈哈哈哈。
```python
        if needle  not in haystack:
            return -1
        return haystack.find(needle)
```
别人:虽然还是用了切片，不过这个方法很直接粗暴。效率更高
```python
    l = len(needle)
    for i in range(len(haystack)-l+1):
        if haystack[i:i+l] == needle:
            return i
    return -1
```
### 35.搜索插入位置
>给定一个排序数组和一个目标值，如果在数组中找到目标值则返回索引。如果没有，返回到它将会被按顺序插入的位置。
你可以假设在数组中无重复元素。

示例
>输入: [1,3,5,6], 5
输出: 2

>输入: [1,3,5,6], 2
输出: 1

>输入: [1,3,5,6], 7
输出: 4

>输入: [1,3,5,6], 0
输出: 0

我
```python
        if target in nums:
            return nums.index(target)
        else:
            if target < nums[0]:
                return 0
            elif target > nums[-1]:
                return len(nums)
            else:
                for i in range(len(nums)-1):
                    if nums[i] < target < nums[i+1]:
                        return i+1
```
总结：查找方式可以采用二分查找提高效率。
别人
```python
        lo = 0
        hi = len(nums)
        while lo < hi:
            mid = lo + (hi - lo) / 2
            if nums[mid] > target:
                hi = mid
            elif nums[mid] < target:
                lo = mid + 1
            else:
                return mid
        return lo
```
### 38数数并说
这题我不知道在说什么，略过。


### 53.最大子序和
描述
>给定一个序列（至少含有 1 个数），从该序列中寻找一个连续的子序列，使得子序列的和最大。
例如，给定序列 [-2,1,-3,4,-1,2,1,-5,4]，
连续子序列 [4,-1,2,1] 的和最大，为 6。

我 v1.0
```python
class Solution:
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        i = 0
        result = nums[0]
        while i < l:
            sums = []
            temp = 0
            for j in range(i, l):
                temp+=nums[j]
                sums.append(temp)
            if result < max(sums):
                result = max(sums)
            i+=1
        return result
```
测试结果如下：
![](https://i.loli.net/2018/04/12/5aced62975043.png)
本地运行时间为14.7s，说明我的方法太粗暴了。应该寻找更好的算法。
![屏幕快照 2018-04-12 上午11.52.43.png](https://i.loli.net/2018/04/12/5aced81b2033c.png)
我 优化后v1.1。优化方案，去掉sums数组，节省空间。但时间复杂度仍然不变。
```python
        l = len(nums)
        i = 0
        result = nums[0]
        while i < l:
            temp = 0
            for j in range(i, l):
                temp+=nums[j]
                if result < temp:
                    result = temp
            i+=1
        return result
```
仍然只通过200/202测试用例，仍然超出时间限制。但本地运行时间为8.3s。有进步。
别人，分治法。时间复杂度O(NlogN)
将输入的序列分成两部分，这个时候有三种情况。
1）最大子序列在左半部分
2）最大子序列在右半部分
3）最大子序列跨越左右部分。
前两种情况通过递归求解。
分治法代码大概如下，emmm。。。目前还没有完全理解。分析可参考https://blog.csdn.net/samjustin1/article/details/52043173
```python

def maxC2(ls,low,upp):  
    #"divide and conquer"  
    if ls is None: return 0  
    elif low==upp: return ls[low]  
      
    mid=(low+upp)/2 #notice: in the higher version python, “/” would get the real value  
    lmax,rmax,tmp,i=0,0,0,mid  
    while i>=low:  
        tmp+=ls[i]  
        if tmp>lmax:  
            lmax=tmp  
        i-=1  
    tmp=0  
    for k in range(mid+1,upp):  
        tmp+=ls[k]  
        if tmp>rmax:  
            rmax=tmp  
    return max3(rmax+lmax,maxC2(ls,low,mid),maxC2(ls,mid+1,upp))  
  
def max3(x,y,z):  
    if x>=y and x>=z:  
        return x  
    return max3(y,z,x)  
```
动态规划算法，时间复杂度为O(n)。
分析：寻找最优子结构。
```python
        l = len(nums)
        i = 0
        sum = 0
        MaxSum = nums[0]
        while i < l:
            sum+=nums[i]
            if sum > MaxSum:
                MaxSum = sum
            if sum < 0:
                sum = 0
            i+=1
        return MaxSum
```
Oh！My god！！！ ！！！！！！！！运行只花了0.2s！！！！！！！！！！！！！！！这也太强了吧！！
![动态规划](https://i.loli.net/2018/04/13/5ad0267adbbc8.png)
优化后,运行时间0.1s.
```python
        sum = 0
        MaxSum = nums[0]
        for i in range(len(nums)):
            sum += nums[i]
            if sum > MaxSum:
                MaxSum = sum
            if sum < 0:
                sum = 0
        return MaxSum
```
其中
`sum += nums[i]`必须紧挨。
```if sum > MaxSum:
                MaxSum = sum
```
### 58.最后一个单词的长度
>描述
给定一个字符串， 包含大小写字母、空格 ' '，请返回其最后一个单词的长度。
如果不存在最后一个单词，请返回 0 。
注意事项：一个单词的界定是，由字母组成，但不包含任何的空格。

>案例
输入: "Hello World"
输出: 5

我
```python
class Solution:
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        l = len(s)
        if l == 0:
            return 0
        n = l-1
        while n >= 0:
            if s[n] == ' ':
                n-=1
            else:
                for i in range(n-1,-1,-1):
                    if s[i] == ' ':
                        return n - i
                return n+1
        return 0
```
别人：使用了split()方法、strip()方法。或者如下
```python
        str = ''
        count = 0
        for i in s[::-1]:
            if str != '' and i==' ':
                return count
            if i != ' ':
                count = count+ 1
                str = str + i
        return count
```
### 66.加一
>描述：给定一个非负整数组成的非空数组，在该数的基础上加一，返回一个新的数组。
最高位数字存放在数组的首位， 数组中每个元素只存储一个数字。
你可以假设除了整数 0 之外，这个整数不会以零开头。

示例1
>输入: [1,2,3]
输出: [1,2,4]
解释: 输入数组表示数字 123。

示例2
>输入: [4,3,2,1]
输出: [4,3,2,2]
解释: 输入数组表示数字 4321。

我
```python
        l = digits[::]
        i = len(digits)-1
        while i >= 0:
            if l[i]+1 <=9:
                l[i]+=1
                return l
            else:           
                if i == 0:
                    l[i] = (l[i]+1)%10
                    l.insert(0,1)
                    return l        
                elif l[i]+1 > 9:
                    l[i]=(l[i]+1)%10
                    if l[i-1]+1 < 10:
                        l[i-1] = l[i-1]+1
                        return l
                    else:
                        i-=1
```
先不说我的代码，先看看别人的，如此干净利落！
```python
        if len(digits) == 0:
            digits = [1]
        elif digits[-1] == 9:
            digits = self.plusOne(digits[:-1])
            digits.extend([0])
        else:
            digits[-1] += 1
        return digits
```
我的思想太粗暴了，题目中说最后一个数要加1.我就老老实实把最后一个元素加一来判断。分两种情况：
1）加1以后小于9。
2）加1以后大于9。但是这个情况又涉及两种情况：a.digits只有一个元素，所以在index=0插入1。b.相加以后产生进位，若将进位加到前面一位数仍产生进位怎么处理。
emmm...我jio得我对步骤的描述并不科学。

大神的就很简洁。题目中也有说digits里的数是1-9，所以只要判断最后一位数是不是9就好。如果是9的话，那个位置就更改为0.而不用向我的一样，再进行求余运算。emmm。。。我把问题的范围扩大了，我的程序可以求解1-18之间的数。这里就两种情况了：最后一位不是9；最后一位是9，那么相加以后会产生进位，就得再接着判断前一位是不是9.这里可以采用循环（像我一样）或者像大神一样（采用递归）。递归的话可以尝试单步调试，有助于理解过程。
### 67.二进制求和
描述
>给定两个二进制字符串，返回他们的和（用二进制表示）。
输入为非空字符串且只包含数字 1 和 0。

示例1
>输入: a = "11", b = "1"
输出: "100"

示例2
>输入: a = "1010", b = "1011"
输出: "10101"

我
```python
        buff = 0
        s = ''
        flag = 1
        flag1 = 1
        l = min(len(a), len(b))
        if len(a) > len(b):
            temp = a[len(a) - l:]
            min_string = b
            other = a[:len(a) - l]
        elif len(a) < len(b):
            temp = b[len(b) - l:]
            min_string = a
            other = b[:len(b) - l]
        else:
            flag1 = 0
            temp = a
            min_string = b
        for i in range(l - 1, -1, -1):
            if buff == 0:
                if temp[i] == min_string[i] == '0':
                    s += '0'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '0'
                    buff = 1
                else:
                    s += '1'
                    buff = 0
            else:
                if temp[i] == min_string[i] == '0':
                    s += '1'
                    buff = 0
                elif temp[i] == min_string[i] == '1':
                    s += '1'
                    buff = 1
                else:
                    s += '0'
                    buff = 1
        if not flag1:
            if buff == 1:
                s += '1'
        elif not buff:
            s += other[::-1]
        else:
            for j in range(len(other) - 1, -1, -1):
                if other[j] == '0':
                    s += '1'
                    if j == 1:
                        s+=other[:1]
                    else:
                        other = other[:j]
                        s += other[::-1]
                    flag = 0
                    break
                else:
                    s += '0'
                    if j == 0:
                        s += '1'
        return s[::-1]
```
代码调了两个多小时，越写越复杂，终于通过。错误主要出现在切片反转。结果一看别人的就两行代码，扎心了！
两行代码
```python
        a, b = int('0b' + a, 2), int('0b' + b, 2)
        return bin(a + b)[2:]
```
emmm...我该说什么呢。哈哈哈让我们看看一行代码的吧！！
```python
        return bin(int(a,2)+int(b,2))[2:]
```
还有另外长一点但比我短的代码，是将a,b反转过来，这样列表切片方便。
int()函数用于将一个字符串或数字转换为整型。语法：
class int(x, base)
x--字符串或数字
base--进制数，默认十进制。
bin()函数返回一个整型int或者长整数long int的二进制表示。
![](https://i.loli.net/2018/04/21/5adb334b9b1b4.png)
bin()运算返回的是二进制。所以前两位是二进制的标志，需要[2:]去除。
### 69.x的平方根
描述
>实现 int sqrt(int x) 函数。
计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例1
>输入: 4
输出: 2

示例2
>输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 由于返回类型是整数，小数部分将被舍去。

我第一想法简单粗暴但超出时间限制。
```python
        if x == 0:
            return 0
        elif x < 0:
            return 'error'
        else:
            i = 1
            while True:
                if (i+1)**2 > x >= i**2:
                    return i
                i+=1
```
改进一下,因为结果肯定是有序的从1，2，3.。。查找，所以使用二分查找。
```python
        if x == 1:
            return 1
        l = 1
        h = x//2
        while h > l:
            m = (l+h)//2
            if m**2 > x:
                h = m-1
            elif m**2 < x:
                l = m+1
            else:
                return m
        if h == l:
            if (h-1)**2 < x < h**2:
                return h-1
            else:
                return h
        elif h < l:
            return h
```
这里比二分查找麻烦的是需要考虑，当结果不在查找的序列是按下取整。这下终于通过了，但是运行时间却是吊车尾。看看别人的，也是用的二分查找，不过是在我的第一版基础上改了一下，不像我这么笨笨地机械套用二分查找。
```python
        if x == 0:
            return 0
        
        l = 1
        r = x
        
        while l <= x:
            res = (l + r) // 2
            s = res**2
            
            if s <= x < (res + 1)**2:
                return res
            if s < x:
                l = res
            if s > x:
                r = res
```
这个问题关键在于结果的下取整。所以我的代码在判断x与m平方的时候可以进行优化。(m-1)**2 <= x < m**2.
### 70.爬楼梯
描述
>假设你正在爬楼梯。需要 n 步你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
注意：给定 n 是一个正整数。

示例1
>输入： 2
输出： 2
解释： 有两种方法可以爬到楼顶。
1.  1 步 + 1 步
2.  2 步

示例2
>输入： 3
输出： 3
解释： 有三种方法可以爬到楼顶。
1.  1 步 + 1 步 + 1 步
2.  1 步 + 2 步
3.  2 步 + 1 步

我第一反应是用递归实现，然而不出所料超出时间限制。
```python
        if n == 1:
            return 1
        elif n == 2:
            return 2
        else:
            s1 = self.climbStairs(n-1)
            s2 = self.climbStairs(n-2)
            return s1+s2 
```
所以。。。。再想一想。既然递归不行就递推试试。emmm...网上说这个问题的结果和Fibonacci数列一样。果然是这样。。。尽管可以转换为求解fibonacci，但是还是应该好好思考一下怎么解决原问题。毕竟可能会联想不到fibonacci。
```python
        nums = [0,1,2]
        if n == 1:
            return nums[1]
        elif n == 2:
            return nums[2]
        else:
            for i in range(3,n+1):
                nums.append(nums[i-1] + nums[i-2])
        return nums[n]
```
上面这个版本通过了，但是只战胜了25%的人，寻求更优的算法吧。
这个很机智啊，我的列表初始化可以优化成如下：
```python
        condition = [0] * (n + 1)
        condition[0] = 1
        condition[1] = 1
        for i in range(2, n+1):
            condition[i] = condition[i-1] + condition[i-2]
        return condition[n]
```
关于循环和递归：
Loops may achieve a performance gain for your program. Recursion may achieve a performance gain for your programmer. Choose which is more important in your situation!
如果使用循环，程序的性能会更高。如果使用递归，程序更容易理解。
### 83.删除排序链表中的重复元素
描述
>给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

示例1
>输入: 1->1->2
输出: 1->2

示例2
>输入: 1->1->2->3->3
输出: 1->2->3

我
```python
        if not head:
            return None
        p = head
        if p.next == None:
            return head
        try:
            while p.next:
                i = p.val
                if p.next.val == i:
                    p.next = p.next.next
                    while p.next:
                        if p.next.val != i:
                            break
                        p.next = p.next.next
                p = p.next
        except AttributeError:
            return head
        else:
            return head
```
写了两三个小时吧，一直改改改。搞得最后逻辑都不是很清楚了，所以引入了异常处理。。。。。然后发现leetcode中输入示例中[1,2,3]头节点是指1.所以返回的是head而不是head.next。
别人的if...else判断一目了然，逻辑很清晰。
```python
class Solution:
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return None
        cur = head
        while cur.next:

            if cur.val == cur.next.val:
                if cur.next.next is None:
                    cur.next = None
                else:
                    temp = cur.next.next
                    cur.next = temp
            else:
                cur = cur.next

        return head
```
### 88.合并两个有序数组
描述
>给定两个有序整数数组 nums1 和 nums2，将 nums2 合并到 nums1 中，使得 num1 成为一个有序数组。
初始化 nums1 和 nums2 的元素数量分别为 m 和 n。
你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。

示例
>输入:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3
输出: [1,2,2,3,5,6]

我
```python
        if n == 0:
            return nums1
        elif n == 0:
            return nums2
        if nums2[-1] <= nums1[0]:
            return nums2 + nums1
        for j in range(len(nums2)):
            if nums2[j] >= nums1[-1]:
                nums1.extend(nums2[j:])
                break
            else:
                l = 0
                h = len(nums1)
                while l < h:
                    m = (l+h)//2
                    if nums2[j] <= nums1[m]:
                        if nums1[m-1] < nums2[j]:
                            nums1.insert(m,nums2[j])
                            break
                        else:
                            h = m-1
                    else: 
                        if nums1[m+1] > nums2[j]:
                            nums1.insert(m+1, nums2[j])
                            break
                        else:
                            l = m+1
        return nums1
```
这么多代码，理所当然超时了呀。。。我的代码丑哭了。并且我并没有假设nums1后面用0来表示nums2占用的空间。思路是，依次遍历nums2中的元素，作为target，然后在nums1中使用二分查找，并插入target。
利用归并排序。
别人：
```python
        end = m + n -1
        m-=1
        n-=1
        while end >= 0 and m >= 0 and n >= 0:
            if nums1[m] >= nums2[n]:
                nums1[end] = nums1[m]
                m-=1
            else:
                nums1[end] = nums2[n]
                n-=1
            end-=1

        while n >= 0:
            nums1[end] = nums2[n]
            n-=1
            end-=1
```
###100.相同的树
描述：
>给定两个二叉树，编写一个函数来检验它们是否相同。
如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

[![示例](https://i.loli.net/2018/04/29/5ae564bc6a702.png)](https://i.loli.net/2018/04/29/5ae564bc6a702.png)

[![](https://i.loli.net/2018/04/29/5ae564dd7cff2.png)](https://i.loli.net/2018/04/29/5ae564dd7cff2.png)

树要遍历每个节点，所以无论怎么写都离不开递归。
```python
        if p == None or q == None:
            return p == q
        elif p.val == q.val:
            return self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
        else:
            return False
```
别人的。。。
```python
        if p == None or q == None:
            return p == q
        return p.val == q.val and self.isSameTree(p.right, q.right) and self.isSameTree(p.left, q.left)
```
### 101.对称二叉树
描述
[![](https://i.loli.net/2018/04/29/5ae5635849d4f.png)](https://i.loli.net/2018/04/29/5ae5635849d4f.png)
我。。。看了别人的跌跌爬爬写出来的。。。
```python
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def f(p, q):
            if p == None:
                return q == None
            if q == None:
                return p == None
            if p.val == q.val:
                return f(p.left, q.right) and f(p.right, q.left)
            if p.val != q.val:
                return False
        if root == None:
            return True
        return f(root.left, root.right)
```
关于二叉树的总结：对于树的操作，是必定要使用递归的方法的。对于求解树的任何问题，都先使用分治法，将树拆分为左子树和右子树（这里主要针对二叉树来说明），如果左子树和右子树都符合条件，则这颗树都满足条件。而左子树和右子树再拆分成左子树和右子树，进行递归。这里需要注意的是递归结束的条件，以及递归过程。
### 104.二叉树的最大深度
描述
[![](https://i.loli.net/2018/04/30/5ae701053c2ad.png)](https://i.loli.net/2018/04/30/5ae701053c2ad.png)
我
```python
        def depth(p):
            if p == None:
                return 1
            else:
                return 1 + max(depth(p.right), depth(q.left))
        
        if root == None:
            return 0
        else:
            return max(depth(root.right), depth(root.left))
```
ok！二十分钟以内一次性通过搞定！！不妄研究了两天的树！！
别人依然两行搞定
```python
        if root == None:
            return 0
        else:
            return 1 + max(maxDepth(root.right), maxDepth(root.left))
```
### 107.二叉树的层次遍历 II
描述
[![](https://i.loli.net/2018/04/30/5ae708055ce4c.png)](https://i.loli.net/2018/04/30/5ae708055ce4c.png)
我又研究了一下树的四种遍历，不然之前学的全忘光了。温故而知新！
```python
    if root is None:
        return []
    p = [root]
    results = []
    current_level_num = 1
    next_level_num = 0
    d = []
    while p:
        current = p.pop(0)
        d.append(current.val)
        current_level_num -= 1
        if current.left != None:
            p.append(current.left)
            next_level_num += 1
        if current.right != None:
            p.append(current.right)
            next_level_num += 1
        if current_level_num == 0:
            current_level_num = next_level_num
            next_level_num = 0
            results.append(d)
            d = []
    return results[::-1]
```
### 108.将有序数组转换为二叉搜索树
描述：
[![](https://i.loli.net/2018/05/02/5ae91c0b21a2f.png)](https://i.loli.net/2018/05/02/5ae91c0b21a2f.png)
我
```python
class Solution:
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        mid = len(nums)//2
        root = TreeNode(nums(mid))
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root
```
卡了我好几天，还特意去学习了二叉搜索树的实现。还有平衡二叉树的插入调整方法。学了半天不得其法。于是看了解答，心里一句mmp。菜鸡就是菜鸡啊。
我忽略了题干的主要条件。数组nums是一个升序数组，所以根节点的位置一定是
m = len(nums)//2 nums中索引值小于m的树都位于根节点的左子树上，而另一半位于根节点的右子树上。
然后进行递归调用。
### 110.平衡二叉树
描述

[![](https://i.loli.net/2018/05/06/5aeed3e9db4b0.png)](https://i.loli.net/2018/05/06/5aeed3e9db4b0.png)

我，两个递归，效率不行。
```python
class Solution:
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        def Depth(root):
            if root is None:
                return 0
            elif root.left i None and root.right is None:
                return 1
            else:
                return 1 + max(Depth(root.left), Depth(root.right))

        if root is None:
            return True
        if abs(Depth(root.left) - Depth(root,right)) > 1:
            return False
        else:
            return isBalanced(root.left) and isBalanced(root.right)
```
我的这个版本运行速度太慢了。看看别人的，直接定义一个求高度的函数。若子树不平衡，则返回-1。
```python
        def height(node):
            if not node:return 0
            left = height(node.left)
            right = height(node.right)
            if left == -1 or right == -1 or abs(left-right) > 1:
                return -1

            return max(left,right) + 1
        return height(root) != -1
```

### 111.二叉树的最小深度
描述
[![](https://i.loli.net/2018/05/03/5aeb27b6c01bf.png)](https://i.loli.net/2018/05/03/5aeb27b6c01bf.png)
我
```python
        if root is None:
            return 0
        p = [root]
        current_level_num = 1
        next_level_num = 0
        i = 1
        while p:
            current = p.pop(0)
            current_level_num-=1
            if current.left is None and current.right is None:
                return i
            if current.left:
                next_level_num+=1
                p.append(current.left)
            if current.right:
                next_level_num+=1
                p.append(current.right)
            if current_level_num == 0:
                i += 1
                current_level_num = next_level_num
                next_level_num = 0
```
我采用的方法是层次遍历，按层次打印结点的，将其稍做修改。用变量i记录当前层数，当遇到叶子结点时则返回当前层数。代码还可以进行优化。待我第二遍再来优化吧，现在需要加快进度。
别人采用递归实现：
```python
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    elif root.left is None:
        return 1 + self.minDepth(root.right)
    elif root.right is None:
        return 1 + self.minDepth(root.left)
    else:
        return 1 + min([self.minDepth(root.left), self.minDepth(root.right)])
```
递归有以下几种情况：
1.根节点为空，深度为0
2.只有一个根节点。深度为1
3.左右子树皆不空，则返回1+左右子树中最小的深度。
4.左子树为空，则返回1+右子树深度。这里可能有点难以理解，可以想象成此时只有根节点a,以及其右子树b，此时最小深度为2。
5.右子树为空，则返回1+左子树深度。同上分析。
[![](https://i.loli.net/2018/05/04/5aec13e7bb9fd.png)](https://i.loli.net/2018/05/04/5aec13e7bb9fd.png)

### 112.路径总和
描述

[![](https://i.loli.net/2018/05/07/5af046091e8ce.png)](https://i.loli.net/2018/05/07/5af046091e8ce.png)

我
```python
class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False
        sum-=root.val
        if sum == 0:
            if root.left is None and root.right is None:
                return True
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
```
其实这个问题我没有完全分析清楚。。。但是代码通过了。。。尴尬。
稍微优化
```python
class Solution:
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False
        if sum == root.val and root.left is None and root.right is None:
            return True
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)
```
### 118.杨辉三角
描述
>给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
在杨辉三角中，每个数是它左上方和右上方的数的和。

示例
>输入: 5
输出:
[
     [1],
    [1,1],
   [1,2,1],
  [1,3,3,1],
 [1,4,6,4,1]
]

我
```python
        if not numRows:
            return []
        p = [1]
        r = [p]
        i = 1
        while i < numRows:
            i+=1
            s = [0] * i
            p = [0] + p + [0]
            for j in range(i):
                s[j] = p[j] + p[j+1]
            r.append(s)
            p = s 
        return r
```
通过了，终于暂时没有树的题了，松了一口气。大神们都用lambda以及map()做的。。。emmm明天来研究。
总结：
1. lambda表达式的用法
+ lambda的主体是一个表达式
+ 例子：
```python
>>> f = lambda x,y: x+y
>>> f(1,2)
3
```
2. zip()函数用法
+ zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同。利用*号操作符，可以将list unzip（解压）。
+ 例子：
```python
>>> a = [1,2,3]
>>> b = [4,5,6]
>>> c = [7,8,9]
>>> d = [4,5]
>>> list(zip(a,b,c))
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
>>> list(zip(a,d))
[(1, 4), (2, 5)]
```
最常应用的地方是矩阵行列互换。
```python
>>> a = [[1,2,3],[4,5,6],[7,8,9]]
>>> list(map(list, zip(*a)))
[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
>>> 
```

3. map()
格式：map(func,seq[,seq2...])
map（）函数是将func作用于seq中的每一个元素，并用一个列表给出返回值。如果func为None，作用同
zip()。(seq表示序列)
例子：
+ 一个seq
```python
>>> list(map(lambda x:x%2, range(4)))
[0, 1, 0, 1]
```

+多个seq，一个返回值
```python
>>> list(map(lambda x,y: x+y,[1,2],[3,4]))
[4, 6]
```
+多个seq，多个返回值组成一个元组
```python
>>> list(map(lambda x,y: (x+y, x*y), [1,2,3],[4,5,6]))
[(5, 4), (7, 10), (9, 18)]
```

4. reduce()
reduce()函数在functools库中。
格式：reduce(func,seq[,init])
reduce()函数即为化简，过程：每次迭代，将上一次的迭代结果（第一次时为init的元素，如果没有init则为seq的第一个元素）与下一个元素一同执行一个二元的func函数。在reduce函数中，init是可选的，如果使用，则作为第一次迭代的第一个元素使用。
reduce(func,[1,2,3]) = func(func(1,2),3)
例子：阶乘
```python
>>> from functools import reduce
>>> reduce(lambda x,y:x*y, [1,2,3,4,5])
120
```
2倍阶乘，初始参数为2。
```python
>>> reduce(lambda x,y:x*y, [1,2,3,4,5],2)
240
```

5.filter()函数
filter()函数用于过滤序列。和map()类似，filter()也接收一个函数和一个序列。和map()不同的是，filter()把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。
例子：删掉偶数
```python
>>> list(filter(lambda x: x%2 == 1, [1,2,3,4,5]))
[1, 3, 5]
```

好，回到正题。
优化后的代码
```python
        r = [[1]]
        for i in range(1,numRows):
            r.append(list(map(lambda x,y:x+y, [0]+r[-1],r[-1]+[0])))
        return r[:numRows]#这里太强了
```
### 119.杨辉三角形 II
描述：
>给定一个非负索引 k，其中 k ≤ 33，返回杨辉三角的第 k 行。

示例：
>输入: 3
输出: [1,3,3,1]

我
```python
class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        p = [1]
        if not rowIndex:
            return p
        for i in range(rowIndex):
            s = list(map(lambda x,y:x+y, [0]+p,p+[0]))
            p = s
        return s
```
运行速度在中下水平。看看最快的。
```python
class Solution:
    def getRow(self, rowIndex):
        """
        :type rowIndex: int
        :rtype: List[int]
        """
        p = [1]
        if not rowIndex:
            return p
        for j in range(rowIndex):
            p = [1] + [p[i]+p[i+1] for i in range(len(p)-1)] +[1]
        return p
```
### 121.买卖股票的最佳时机
描述
>给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。
注意你不能在买入股票前卖出股票。

示例1
>输入: [7,1,5,3,6,4]
输出: 5
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格。

示例2
>输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

我
```python
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        l = []
        for i in range(len(prices)):
            for j in range(i+1, len(prices)):
                if prices[j] > prices[i]:
                    l.append(prices[j]-prices[i])
        if not l:
            return 0
        else:
            return max(l)
```
果不其然，超出时间限制。两层循环
然后修改了代码，表面上看只有一层循环，但事实上本质还是两层循环。虽然比第一版本的运行时间要快一点，但是依然超出时间限制。
```python
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if not prices:
            return 0
        l = []
        for i in range(len(prices)-1):
            j = max(prices[i+1:])
            if j > prices[i]:
                l.append(j-prices[i])
        if not l:
            return 0
        else:
            return max(l)
```
看看别人的思路。在价格最低的时候买入，差价最大的时候卖出就解决了！没毛病这个方案，跟实际买卖很像，然而真的炒股的时候我们并不能预测每一天的价格。
```python
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if len(prices) < 2:
            return 0
        profit = 0
        minimum = prices[0]
        for i in prices:
            minimum = min(i, minimum)
            profit = max(i - minimum, profit)
        return profit
```
一次循环搞定查找最小值和最大差值。
### 122.买卖股票的最佳时机II
描述
>给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。
设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

示例1
>输入: [7,1,5,3,6,4]
输出: 7
解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

示例2
>输入: [1,2,3,4,5]
输出: 4
解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
     注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
     因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

示例3
>输入: [7,6,4,3,1]
输出: 0
解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

我
```python
        profit = 0
        i = 0
        while i < len(prices)-1:
            minimum = prices[i]
            j = i+1
            if prices[i] >= prices[j]:
                minimum = min(prices[j], minimum)
                i+=1
            else:
                while j <= len(prices)-1:
                    if j == len(prices)-1:
                        profit+=prices[j] - minimum
                        return profit
                    elif prices[j] <= prices[j+1]:
                        j+=1
                    else:
                        profit+=prices[j]-minimum
                        i = j+1
                        break
        return profit
```
终于又靠自己通过了，这感觉真爽。
看看别人的。真的强，逻辑。
```python
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for i in range(1,len(prices)):
            if prices[i] > prices[i-1]:
                profit += prices[i]-prices[i-1]
        return profit
```
比如[1,3,5,2,6,1,3]
profit=（3-1)+(5-3)+(6-2)+(3-1)=3-1+5-3+(6-2)+(3-1)=(5-1)+(6-2)+(3-1)
而我的是直接求解profit = (5-1)+(6-2)+(3-1)
### 125.验证回文串
描述
>给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
说明：本题中，我们将空字符串定义为有效的回文串。

示例1
>输入: "A man, a plan, a canal: Panama"
输出: true

示例2
>输入: "race a car"
输出: false

我
```python
        if not s:
            return True
        def f(i):
            q = i.isalpha()
            if 48<=ord(i)<=57:
                p = 1
            else:
                p = 0
            return q or p
        l = list((filter(f, s)))
        s = ''.join(l)
        s = s.upper()
        return s[::] == s[::-1]
```
通过了，不过借助了外力。还是查询了如何判断字符串中的字母。使用str.isalpha()方法判断，而数字则使用ASCII码判断(其实这里可以用str.isdigit()来判断数字)python真的太高级。然后前天总结的filter()函数派上用场了哈哈哈。对于字符串转列表，列表转字符串的用法还需要再总结，每次使用都要现查，浪费时间。题目中还需要将字符统一转化为大写或小写。
别人的，两句话搞定。。。这大概就是刷题的意义吧。开开心心地终于通过了，然后发现别人只用两三行代码就解决了问题，啊哈哈哈哈，酸爽。
```python
        s = list(filter(str.isalnum, s.lower()))
        return True if s == s[::-1] else False
```
总结：str.isalnum()方法检测字符串是否由字母和数字组成。
字符串转列表：
str.split('')
列表转字符串：
s = ''.join(l)
### 136.只出现一次的数字
描述
>给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

>示例 1:
输入: [2,2,1]
输出: 1

>示例 2:
输入: [4,1,2,1,2]
输出: 4

我v1.0
```python
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        s = []
        for i in nums:
            if i not in s:
                s.append(i)
            elif i in s:
                s.remove(i)
        return s[0]
```
很好，超出时间限制。早已在掌握之中。
下面的v2.0版本是为了缩减代码，但效率上没有任何改善。
```Python
        for i in nums:
            if nums.count(i) == 1:
                return i
```
看了别人的代码，发现我第一个版本的思路还不错，只是将列表s换成字典会好很多。
```python
        s = {}
        for i in nums:
            if i in s.keys():
                s.pop(i)
            else:
                s[i]=1
        return list(s.keys())[0]
```
很好通过了，但是执行时间还有可提高空间下面看一看终极boss。
高级用法异或^
0异或任何数不变，任何数与自己异或为0。a⊕b⊕a=b。异或满足加法结合律和交换律。
666异或操作真的很强，我的提交执行用时战胜了95.75%的Python3提交记录。
v3.0
```python
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for i in nums:
            res^=i
        return res
```
### 141.环形链表
描述
>给定一个链表，判断链表中是否有环。
进阶：
你能否不使用额外空间解决此题？

我...遍历了以后超出时间限制，于是看大家总结的方法。一个就是设置两个指针slow和fast，一个步长为1，一个步长为2进行遍历。如果有环，则slow和fast总会在某一点相遇。如果没有环，则fast会先为空，或者fast.next为空。
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None or head.next.next is None:
            return False
        slow = head.next
        fast = head.next.next
        while slow != fast and fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
        if fast == slow:
            return True
        else:
            return False
```
通过了呢，但是我觉得我的代码不够优雅，还可以再进行简化。
关于环形链表的相关问题可以[查看](https://blog.csdn.net/happywq2009/article/details/44313155)
```python
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        fast = slow = head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if slow == fast:
                return True
        return False
```
关于环形链表问题的[总结（from Rotten_Pencil）](https://www.jianshu.com/p/1c59b153308c)
1.快慢指针一直到相遇时的循环次数等于环的长度。（可推导）
Case1:一个完美的环状链表，即链表头尾相连
>一个环形链表：{A，B，C，A，B，C，……}
其上存在两个指针，A指针移动速度是B指针的两倍。
A，B同时从节点1出发，所经过的节点如下：
快指针A：A->C->B->A
慢指针B：A->B->C->A
A、B指针在节点A第一次相遇，循环次数为3，而环的程度正好也为3。那这个是不是巧合呢？
首先我们要理解的是循环的次数代表的是什么。
1. 每次循环，对于B这个慢指针来说，意味着走了一个单位长度。
2. 而对于A来说，走了两个单位长度。
3. 那么二者第一次相遇必然是在A走了2圈，B走了1圈的时候。
4. 假如A的速度是B的3倍，那么二者第一次相遇是在A走了3圈，B走了1圈的时候。
5. 同理A是B的5倍速度，相遇时A走了5圈，B走了1圈
...
n. A的速度是B的n倍，相遇时A走了n圈，B走了1圈
从上面的观察我们可以发现，无论A的速度是B的几倍，两者第一次相遇必然是在B走了1圈时。
因为B的速度代表的是链表基本的长度单位，即从一个节点移动到下一个节点的距离。
同时在链表中，每个节点与节点之间这个距离是不变的。
当循环结束时，B走了1圈，正好是环的长度。而B每次移动一个单位距离，因此环的长度等于循环次数。

Case2：不完美的环状链表，即，链表中某一中间节点与尾部相连
![不完美的环形链表](https://upload-images.jianshu.io/upload_images/2206027-675224c588ace55d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/397)

>一个环形链表（如图所示）：{D，E，A，B，C，A，B，C，……}
其上存在两个指针，A指针移动速度是B指针的两倍。
A，B同时从节点1出发，所经过的节点如下：
快指针A：D->A->C->B
慢指针B：D->E->A->B
根据上图，我们可以计算出A、B行走的距离：
A = d+e+a+b+c+a
B = d+e+a
因为A的速度是B的2倍，那么A行走的距离也因该是B的2倍：
d+e+a+b+c+a = 2(d+e+a)
      a+b+c = d+e+a
从上图可以看出，a+b+c正好是环的长度，而d+e+a则是B行进的距离。
又知，每次循环B移动一个单位距离，因此在不完美的环状表中，循环次数亦是等于环的长度。

2.快慢指针相遇点到环入口的距离 = 链表起始点到环入口的距离。（可推导）
>根据上文公式，我们可以继续推导，即：
a+b+c = d+e+a
  b+c = d+e
b+c是相遇点到环入口的距离
d+e是链表起点到环入口的距离

[相关问题](https://blog.csdn.net/happywq2009/article/details/44313155)：
- 判断是否为环形链表
思路：使用追赶的方法，设定两个指针slow、fast，从头指针开始，每次分别前进1步、2步。如存在环，则两者相遇；如不存在环，fast遇到NULL退出。
- 若为环形链表，求环入口点
思路：快慢指针相遇点到环入口的距离 = 链表起始点到环入口的距离
- 求环的长度
思路：记录下相遇点p，slow、fast从该点开始，再次碰撞所走过的操作数就是环的长度s
- 判断两个链表是不是相交（思路：如果两个链表相交，那么这两个链表的尾节点一定相同。直接判断尾节点是否相同即可。这里把这道题放在环形链表，因为环形链表可以拆成Y字的两个链表。）

### 142.环形链表 II
描述(虽然这是中等难度的题，不过我觉得有必要跟上一题放在一起。我可以考虑按类型来做题，不再按难度来做题了)：
>
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
说明：不允许修改给定的链表。
进阶：
你是否可以不用额外空间解决此题？

我
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return None
        fast = head
        slow = head
        while fast.next and fast.next.next:
            slow = slow.next
            i+=1
            fast = fast.next.next
            if slow == fast:
                p = head
                while slow != p:
                    p = p.next
                    slow = slow.next
                return p
        return None
```
思路就是上面总结的：相遇点到环入口点的距离=头节点到环入口点的距离

### 155.最小栈
描述：
>
设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。
push(x) -- 将元素 x 推入栈中。
pop() -- 删除栈顶的元素。
top() -- 获取栈顶元素。
getMin() -- 检索栈中的最小元素。

示例
>MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.getMin();   --> 返回 -2.

我(最近在忙项目的事，好久没刷题了，哈哈哈现在告一段落又开始刷题，一遍过真的爽嘻嘻)
```python
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.l = []

        
    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if x is None:
            pass
        else:
            self.l.append(x)
        

    def pop(self):
        """
        :rtype: void
        """
        if self.l is None:
            return 'error'
        else:
            self.l.pop(-1)
        

    def top(self):
        """
        :rtype: int
        """
        if self.l is None:
            return 'error'
        else:
            return self.l[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        if self.l is None:
            return 'error'
        else:
            return min(self.l)
```
看看执行速度快的代码
```python
执行用时为 60 ms 的范例
class MinStack(object):
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = None

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        if self.min == None or self.min > x:
            self.min = x

    def pop(self):
        """
        :rtype: void
        """

        popItem = self.stack.pop()
        if len(self.stack) == 0:
            self.min = None
            return popItem

        if popItem == self.min:
            self.min = self.stack[0]
            for i in self.stack:
                if i < self.min:
                    self.min = i
        return popItem

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.min
```
__init__()初始化的时候多定义了一个最小值self.min,在每次push和pop操作的时候就判断是否为最小值，保证self.min是最小的。这个运行起来比我用min()要快好多。也更能体现刷题的意义，2333.

### 160.相交链表
描述
>[![](https://i.loli.net/2018/05/17/5afd9f8d120aa.png)](https://i.loli.net/2018/05/17/5afd9f8d120aa.png)

2333这个题之前有总结到啊。
思路是这样的（题目中假设没有环）：
1.分别遍历两个链表，如果尾节点不同则不相交，返回None，如果尾节点相同则求相交结点。
2.求相交结点的方法是，求出链表长度的差值，长链表的指针先想后移动lenA-lenB。然后两个链表一起往后走，若结点相同则第一个相交点。
3.求链表的长度，在遍历的时候就计算，并将每个结点放在字典中。
该题中不让修改链表结构。所以只考虑以上思路。还有另一种方法是：
先遍历第一个链表到他的尾部，然后将尾部的next指针指向第二个链表(尾部指针的next本来指向的是null)。这样两个链表就合成了一个链表，判断原来的两个链表是否相交也就转变成了判断新的链表是否有环的问题了：即判断单链表是否有环？
我
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a = headA
        b = headB
        i = 0
        l = m =0
        while a:
            a = a.next
            l += 1
        while b:
            b = b.next
            m += 1
        if a != b:
            return None
        a = headA
        b = headB        
        if l > m:
            diff = l - m
            while diff:
                a = a.next
                diff-=1
        if l < m:
            diff = m - l
            while diff:
                b = b.next
                diff-=1
        while a!=b:
            a = a.next
            b = b.next
        return a
```
优化一下：上个版本的代码用了字典存每个结点，但最后判断的时候却不知道用，依然采用了a=a.next的方法访问下一个结点。所以以下代码将改为根据结点的顺序为键，查找字典里的结点，大大提高了运行速度。但这是使用空间来换时间的做法。
优化
```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        a = headA
        b = headB
        da = {0:a}
        db = {0:b}
        i = 0
        l = m =0
        while a:
            a = a.next
            l += 1
            da[l] = a
        while b:
            b = b.next
            m += 1
            db[m]  = b
        if db[m] != da[l]:
            return None  
        i = 0
        if l >= m:
            diff = l - m 
            while True:
                if da[i]==db[diff]:
                    return da[i]
                diff+=1
                i+=1
        if l < m:
            diff = m - l
            while True:
                if db[i] == da[diff]:
                    return db[i]
                diff+=1
                i+=1
```
2333，别人的代码又少又快，这就是差距。简单的问题复杂化。
```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        p1 = headA
        p2 = headB
        while(p1 != p2):
            p1 = headB if p1 == None else p1.next
            p2 = headA if p2 == None else p2.next
        return p1
```
再看看排第二的代码
```python
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        lenA, lenB = 0, 0
        pA = headA
        pB = headB
        while pA:
            pA = pA.next
            lenA += 1
        while pB:
            pB = pB.next
            lenB += 1
        pA = headA
        pB = headB
        if lenA > lenB:
            for i in range(lenA-lenB):
                pA = pA.next
        else:
            for i in range(lenB-lenA):
                pB = pB.next
        while pA!=pB:
            pA = pA.next
            pB = pB.next
        return pA
```
思路跟我的几乎一样，只是for循环替代了while。这真是能用for就别用while。也不是，leetcode相同的代码提交多次执行速度都不一样。。。不稳定，仅供参考。
总结：
在本题中，求链表相交的时候假设没有环存在。但实际中，链表相交有多种情况。[参考](https://www.cnblogs.com/cwkpql/p/4743602.html)
1.A无环，B无环，存在以下三种情况：
[!](https://images0.cnblogs.com/blog2015/792106/201508/192307273635528.png)
2.A有环，B无环(A无环，B有环亦然)，存在一种情况：
[!](https://images0.cnblogs.com/blog2015/792106/201508/192309478784489.png)
3.A有环，B有环，则存在三种情况：
[!](https://images0.cnblogs.com/blog2015/792106/201508/192310476138505.png)
判断一个单向是否有环的方法：从该链表头指针开始，设置两个指针，一个快指针，一个慢指针。快指针每次沿着链表走两个链节，慢指针每次沿着链表走一个链节，当两个指针指向同一个链节时（即两个指针相等），说明链表有环；当快指针指向null时，说明链表无环。

对于情况1：
　　　　分别遍历得到两个链表的长度，并且在遍历的时候，比较两个链表的终节点，如果不同，则是(2)；如果相同，则为(1)或(3).对(1)(3),取链表节点数的差值，将长链表从差值处开始遍历，短节点从头结点开始遍历，比较两个节点是否相等，不相等继续向下遍历，直至相等找到相交节点。

对于情况2：
　　　　直接得出结论，二者不相交。

对于情况3：
　　　　判断A有环的因为快慢指针指向了同一个节点，记为交点A，B的记为交点B。从交点A放出一个慢指针，每次沿链表走一个链节；交点B放出一个快指针，每次沿着链表走两个链节。当两个指针相交，则说明A、B共用一个环，情形为(2)或(3)；当A的慢指针走回交点A，两个指针仍然没相交，说明情况为(1).对(2)(3),将交点A的next指向null，断开环路然后按照1(3)中的线性查找方法去找相交节点，一定能找到一个交点C。我们将交点A的next复原，从交点C开始遍历，找到交点C的前节点，让交点C前节点的next指向null，这个时候能够找到交点D。当交点C与交点D相同时，为情况(2);不相等时为情况(3).


### 167.两数之和 II - 输入有序数组
描述
>给定一个已按照升序排列 的有序数组，找到两个数使得它们相加之和等于目标数。
函数应该返回这两个下标值 index1 和 index2，其中 index1 必须小于 index2。
说明:
返回的下标值（index1 和 index2）不是从零开始的。
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例
>输入: numbers = [2, 7, 11, 15], target = 9
输出: [1,2]
解释: 2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

我
```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        s = []
        r = []
        for i in range(len(numbers)):
            if numbers[i] in s:
                r.append(s.index(numbers[i])+1)
                r.append(i+1)
                return r
            s.append(target-numbers[i])
        return None
```
超出时间限制。开始上下求索。很好，马上将列表换成字典，效率马上上去了，一下就通过，基于索引表实现的字典果然很强。
优化后通过
```python
class Solution(object):
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        s = {}
        r = []
        for i in range(len(numbers)):
            if numbers[i] in s.keys():
                r.append(s[numbers[i]]+1)
                r.append(i+1)
                return r
            s[target-numbers[i]] = i
        return None
```

### 168.Excel表列名称
描述
>给定一个正整数，返回它在 Excel 表中相对应的列名称。
例如，
    1 -> A
    2 -> B
    3 -> C
    ...
    26 -> Z
    27 -> AA
    28 -> AB 
    ...
示例
>输入: 1
输出: "A"

>输入: 28
输出: "AB"

>输入: 701
输出: "ZY"

我
```python
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        d = {}
        r = []
        a = ''
        for i in range(1,27):
            d[i] = chr(64+i)
        if n <= 26:
            return d[n]
        if n % 26 == 0:
            n = n/26 - 1
            a ='Z'
        while n > 26:
            s = n % 26
            n = n // 26
            r.append(s)
        result = d[n]
        for i in r[::-1]:
            result+=d[i]
        return result + a
```
通过了leetcode，但是自己测试，发现代码并不对。输入值为x乘以26，其中x-1可以被26整除时，我的程序就没法运行。hhh万万没想到居然贡献了一个测试用例。
别人。将十进制转换为二十六进制，但是从1开始，所以需要减一个1。emmm...涉及到数学推导的我也理解很慢，需要请教一下别人加深理解。
```python
class Solution(object):
    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        #需要注意26时：26%26为0 也就是0为A 所以使用n-1  A的ASCII码为65
        result = ""
        while n != 0:
            result = chr((n-1)%26+65) + result
            n = (n-1)/26
        return result
```
总结一下：
字符与ASCII码的转换：
- 字符转ASCII码 ord(str)，如ord('A')为65
- ASCII码转字符 chr(int)，如chr(65)为'A'

### 169.求众数
描述
>给定一个大小为 n 的数组，找到其中的众数。众数是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
你可以假设数组是非空的，并且给定的数组总是存在众数。

示例
>输入: [3,2,3]
输出: 3
输入: [2,2,1,1,1,2,2]
输出: 2

我，题目要求输出的只是一个数，但其实众数不是唯一的，我觉得应该输出数组。。。。
```python
        d = {}
        r = []
        for i in range(len(nums)):
            if nums[i] in d.keys():
                d[nums[i]]+=1
            else:
                d[nums[i]] = 1
        times = len(nums)//2
        for key in d.keys():
            if d[key] > times:
                return key
```

大神一句话的事！省题很重要，题目中假设了每个数组都有众数。
```python
class Solution(object):
    def majorityElement(self, num):
        return sorted(num)[len(num)/2]
```
总结：求众数先排序，取中间值。
还有一个版本，我觉得max()函数的使用值得借鉴,找出数组中值最大的那组数据。如果是max(dict)，只能找出键最大的那组数据，所以这个时候就得使用max()函数中的key。
```python
        num_dict = {}
        for i in nums:
            if i in num_dict:
                num_dict[i] +=1
            else:
                num_dict[i] = 1
        return max(num_dict.items(), key=lambda x: x[1])[0]
```
平时使用max/min()函数都是传入一组数，但其实完整的函数定义是
max(iterable,key,default).s求迭代器的最大值，其中iterable 为迭代器，max会for i in … 遍历一遍这个迭代器，然后将迭代器的每一个返回值当做参数传给key=func 中的func(一般用lambda表达式定义) ，然后将func的执行结果传给key，然后以key为标准进行大小的判断。
```python
d1 = {'name': 'egon', 'price': 100}
d2 = {'name': 'rdw', 'price': 666}
d3 = {'name': 'zat', 'price': 1}
l1 = [d1, d2, d3]
a = max(l1, key=lambda x: x['name'])
print(a)
b = max(l1, key=lambda x: x['price'])
print(b)
```
执行结果：
{'name': 'zat', 'price': 1}
{'name': 'rdw', 'price': 666}
关于max()/min()函数的使用，这遍[博文](https://www.cnblogs.com/whatisfantasy/p/6273913.html)不错。
啰嗦一句题外话：
```
>>> prices
{'A': 123, 'B': 450.1, 'C': 12, 'E': 444}
>>> list(zip(prices.values(),prices.keys()))
[(123, 'A'), (450.1, 'B'), (12, 'C'), (444, 'E')]
```


### 171. Excel表列序号
描述
>给定一个Excel表格中的列名称，返回其相应的列序号。
例如，
    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 
    ...

示例
>输入: "A"
输出: 1
输入: "AB"
输出: 28
输入: "ZY"
输出: 701

我
```python
class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        num = 0
        for i in range(len(s)-1,-1,-1):
            num += (ord(s[i]) - 64) * 26**(len(s)-i-1)
        return num
```
hhhhhhhhh这道题有了上次的基础，第一次提交执行用时战胜了100.00%的提交记录。完美。就是将二十六进制转换为十进制，初始值是1。这次终于可以不看别人的解答了！！！

### 172.阶乘后的零
描述
>给定一个整数 n，返回 n! 结果尾数中零的数量。

示例1
>输入: 3
输出: 0
解释: 3! = 6, 尾数中没有零。

示例2
>输入: 5
输出: 1
解释: 5! = 120, 尾数中有 1 个零.

说明: 你算法的时间复杂度应为 O(log n) 。
[分析](https://www.cnblogs.com/hutonm/p/5624996.html)
我的思考过程是这样的：
1）如果将阶乘结果计算出来的话，求零可以采用能不能整出10的倍数来看，或者将结果转为字符串，利用str.count()求'0'的个数。
2）但是计算阶乘要不用递归要不用递推，运行时间都要很长，无法满足题目中的logn要求。
3）于是想到求0也就是求其中2*5的个数，也就是5的个数，因为每一个偶数都含2，只要有5肯定有2。然后我就遍历求能不能被5整除了。。。好蠢。。。还遍历
4）正确的做法是使用n除以5。。。具体的话上述链接分析得很好。

```python
class Solution(object):
    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = 0
        while n >= 5:
            n = n // 5
            r+=n
        return r
```

### 189.旋转数组
描述
>
给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

示例1
>输入: [1,2,3,4,5,6,7] 和 k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

示例2
>输入: [-1,-100,3,99] 和 k = 2
输出: [3,99,-1,-100]
解释: 
向右旋转 1 步: [99,-1,-100,3]
向右旋转 2 步: [3,99,-1,-100]

说明
>尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
要求使用空间复杂度为 O(1) 的原地算法。

```python
# coding:utf8
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        #方法1
        '''def rev(start, end , s):
            while start < end:
                s[end] ,s[start]= s[start], s[end]
                end-=1
                start+=1
        length = len(nums)
        rev(length-k%length,length-1,nums)
        rev(0,length-k%length-1,nums)
        rev(0,length-1,nums)'''
        #方法2
        '''
        l = len(nums)
        nums[:] = nums[l-k:] + nums[:l-k]'''
        #方法3
        l = len(nums)
        nums[:l-k] = reversed(nums[:l-k])
        nums[l-k:] = reversed(nums[l-k:])
        nums[:] = reversed(nums)
```
### 190.颠倒二进制位
描述
>颠倒给定的 32 位无符号整数的二进制位。

示例
>输入: 43261596
输出: 964176192
解释: 43261596 的二进制表示形式为 00000010100101000001111010011100 ，
     返回 964176192，其二进制表示形式为 00111001011110000010100101000000 。

进阶
>如果多次调用这个函数，你将如何优化你的算法？

我,ide不出错，但leetcode不通过。
```python
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        l = list(bin(n))[2:]
        l = list(map(int,l))
        l.reverse()
        l = list(map(str,l))    
        l = ['0','b'] + l
        s = ''.join(l)
        return int(s,2)
```
看到网上说需要格式化为32位无符号数。
即使用python的format格式化。'{0:032b}'.format(n)将n转化为32位无符号数。
修改
```python
class Solution:
    # @param n, an integer
    # @return an integer
    def reverseBits(self, n):
        l = list('{0:032b}'.format(n))
        l = list(map(int,l))
        l.reverse()
        l = list(map(str,l))
        s = ''.join(l)
        return int(s,2)
```
二刷
```python
        b = list('{:032b}'.format(n))
        for i in range(16):
            b[i],b[31-i] = b[31-i],b[i]
        r = int(''.join(b),2)
        return r
```
然后其实大家都使用的是移位运算来处理。format和移位运算已做出总结。
### 191.位1的个数
描述：
>编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。

示例
>输入: 11
输出: 3
解释: 整数 11 的二进制表示为 00000000000000000000000000001011
输入: 128
输出: 1
解释: 整数 128 的二进制表示为 00000000000000000000000010000000

我
```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        r = 0
        l = list(map(int,list('{:b}'.format(n))))
        for i in l:
            r += i
        return r
```
第一
```python
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        n=bin(n)
        return n.count('1')
```
### 198.打家劫舍
描述
>
你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

示例
>输入: [1,2,3,1]
输出: 4
解释: 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
输入: [2,7,9,3,1]
输出: 12
解释: 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
     偷窃到的最高金额 = 2 + 9 + 1 = 12 。

我,虽然我知道要用动态规划，不过一下子想不出来动态规划怎么来，于是先一步步来，首先写个递归
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums)
        else:
            q = self.rob(nums[:-1])
            p = self.rob(nums[:-2])
            return max(q,p+nums[-1])
```
然后超时。
第二版 递推，通过
```python
        r = {}
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0],nums[1])
        else:
            r[0] = nums[0]
            r[1] = max(nums[0],nums[1])
            for i in range(2,len(nums)):
                r[i] = max(r[i-1],r[i-2]+nums[i])
        return r[len(nums)-1]
```
别人的
```python
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        last = 0 
        now = 0
        for i in nums: last, now = now, max(last + i, now)
        return now
```

### 202.快乐数
描述
>编写一个算法来判断一个数是不是“快乐数”。
一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

示例
>输入: 19
输出: true
解释: 
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

我
题目中有说，结果有两种情况：
1. 结果为1则为true;
2. 结果无限循环，这就需要保存计算过的值，当然是使用散列表实现的字典。这里如果使用递归的话很难维护字典，所以最后改为递推。
```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        d = {}
        while True:
            l = list(map(int,list(str(n))))
            m = 0
            for i in l:
                m += i**2
            if m in d:
                print(d)
                return False
            if m == 1:
                print(d)
                return True
            d[m] = m
            n = m
```
上个版本中，取每个位置上的数时用了python的特有方法。但如果使用数学方法会跟高效。
优化
```python
        d = {}
        while True:
            m = 0
            while n > 0:
                m += (n%10)**2
                n //= 10 
            if m in d:
                return False
            if m == 1:
                return True
            d[m] = m
            n = m
```
但其实效果不是很明显，还可以更优化。看看别人的：
```python
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        record = []
        sq_sum = 0
        se_n = n
        
        while se_n != 1:
            sq_sum = 0
            while se_n > 0:
                sq_sum += (se_n % 10) * (se_n % 10)
                se_n = se_n / 10
            if sq_sum in record:
                return False
            record.append(sq_sum)
            se_n = sq_sum
            
        return True
```
emmm...大概思路是一样的，但是逻辑判断顺序不一样，所以在时间上有区别。
### 203.删除链表中的节点
描述
>删除链表中等于给定值 val 的所有节点。

示例
>输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5

我
```python
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        if head:
            while head.val == val:
                head = head.next
                if head is None:
                    return head
            q = head
            p = q.next
            while p:
                if p.val == val:
                    q.next = p.next
                else:
                    q = q.next
                p = p.next
        return head
```
leetcode中所有涉及到链表的题，头节点都保存值，并不是我们通常认为的头节点不存储值，所以以上代码中while循环用来考虑第一个节点的值。但其实最好的办法是没有头节点我们给它创造一个头节点。
```python
        pre_node = ListNode(None)
        pre_node.next = head
        q = pre_node
        p = q.next
        while p:
            if p.val == val:
                q.next = p.next
            else:
                q = q.next
            p = p.next
        return pre_node.next
```
再次优化
```python
        pre_node = ListNode(None)
        pre_node.next = head
        q = pre_node
        while q.next:
            if q.next.val == val:
                q.next = q.next.next
            else:
                q = q.next
```
### 204.计数质数
描述
>统计所有小于非负整数 n 的质数的数量。

示例
>输入: 10
输出: 4
解释: 小于 10 的质数一共有 4 个, 它们是 2, 3, 5, 7 。

我
```python
        import math
        flag = 0
        sum = 0
        for i in range(2, n):
            for j in range(2, int(math.sqrt(i))+1):
                if i % j == 0:
                    flag = 1
                    break
            if flag == 0:
                sum += 1
            else:
                flag = 0
        return sum
```
依次遍历超时，没办法了。
#### 厄拉多塞筛法
>西元前250年，希腊数学家厄拉多塞(Eeatosthese)想到了一个非常美妙的质数筛法，减少了逐一检查每个数的的步骤，可以比较简单的从一大堆数字之中，筛选出质数来，这方法被称作厄拉多塞筛法(Sieve of Eeatosthese)。
具体操作：先将 2~n 的各个数放入表中，然后在2的上面画一个圆圈，然后划去2的其他倍数；第一个既未画圈又没有被划去的数是3，将它画圈，再划去3的其他倍数；现在既未画圈又没有被划去的第一个数是5，将它画圈，并划去5的其他倍数……依次类推，一直到所有小于或等于n的各数都画了圈或划去为止。这时，表中画了圈的以及未划去的那些数正好就是小于 n 的素数。

使用此方法以后太高级了！
```python
def countPrime(n):
    if n < 3:
        return 0
    prime = [1] * n
    prime[0] = prime[1] = 0
    for i in range(2, int(n**0.5) +1):
        if prime[i] == 1:
            prime[i*i:n:i] = [0]*len(prime[i*i:n:i])
    return sum(prime)
```
题外话，Python即对象，无意中看到觉得挺有意思的，放在这里。
#Python 列表操作 a[:10] = [x for x in range(100)] 实际上发生了什么？ 后面的 90 个元素怎么处理的？
```
>>> a = list() 
>>> a = [chr(_) for _ in range(ord('a'), ord('z')+1)] 
>>> a 
['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', ' y', 'z'] 
>>> a[:10] = [x for x in range(100)] 
>>> a 
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 6 3, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
```

### 205.同构字符串
描述
>给定两个字符串 s 和 t，判断它们是否是同构的。
如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。
所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

示例
>输入: s = "egg", t = "add"
输出: true
输入: s = "foo", t = "bar"
输出: false
输入: s = "paper", t = "title"
输出: true

说明:
你可以假设 s 和 t 具有相同的长度。

分析一波：
1）先判断两个字符串的格式匹不匹配，即是否同为'ABB'等格式，若格式不相同则返回False，否则进入第二步；
2）判断是否存在不同字符映射到同一字符的情况，若存在，则False。这条判断通过字典来实现，其中键为被映射的字符。比如'eggdd'--->'addaa'为False。在判断的时候注意不能仅通过该键值是否存在，因为会有'paper'--->'title'的情况存在，因为'p'-->'t'存在多次映射，所以还需要判断dict[t[i]] == s[i] ?

我
```python
class Solution:
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        l = len(s)
        if l == 0 or l == 1:
            return True
        d = {t[0]:s[0]}
        for i in range(1, l):
            if s[i] == s[i-1]:
                if t[i] == t[i-1]:
                    continue
                else:
                    return False
            else:
                if t[i] in d:
                    if d[t[i]] == s[i]:
                        continue
                    else:   
                        return False
                else:
                    d[t[i]] = s[i]
        return True
```
看看别人的提神醒脑：
```python
        return len(set(zip(s,t))) == len(set(s)) == len(set(t))
```
太高级，over!

### 290.单词模式
顺便把相关的290给做了。
描述
>
给定一种 pattern(模式) 和一个字符串 str ，判断 str 是否遵循相同的模式。
这里的遵循指完全匹配，例如，pattern里的每个字母和字符串str中的每个非空单词之间存在着双向连接的对应模式。

示例
>输入: pattern = "abba", str = "dog cat cat dog"
输出: true
输入:pattern = "abba", str = "dog cat cat fish"
输出: false
输入: pattern = "aaaa", str = "dog cat cat dog"
输出: false
输入: pattern = "abba", str = "dog dog dog dog"
输出: false

说明
>你可以假设 pattern 只包含小写字母， str 包含了由单个空格分隔的小写字母。  

我
```python
class Solution:
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        l = str.split(' ')
        if len(l) != len(pattern):
            return False
        return len(set(zip(pattern,l))) == len(set(pattern)) == len(set(l))
```
直接使用了上述的方法，只是这题需要注意的是题目并没有说pattern与str等长，所以要额外判断pattern与str中单词数是否相等。
不然以下用例会出错输入: 
pattern = "abba", str = "dog dog dog dog"
输出: false

### 206.反转链表
描述
>反转一个单链表

示例
>输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

我
```python
class Solution:
        def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        p = head
        d = {}
        i = 0
        while p:
            d[i] = p
            p = p.next
            i += 1
        l = len(d)
        for i in range(l-1,0,-1):
            d[i].next = d[i-1]
        d[0].next = None
        return d[l-1]

```

另外一种解法
```python
        if head is None:
            return None
        cur = head
        pre = None
        nxt = cur.next
        while nxt:
            cur.next = pre
            pre = cur
            cur = nxt
            nxt = nxt.next
        cur.next = pre
        head = cur
        return head
```

### 234.回文链表
描述
>请判断一个链表是否为回文链表

示例
>输入: 1->2
输出: false

输入: 1->2->2->1
输出: true

进阶：你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

我
```python
class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return True
        l = []
        p = head
        while p.next:
            l.append(p.val)
            p = p.next
        l.append(p.val)
        return l == l[::-1]
```
但是空间占用太大。另一种解法空间复杂度O(1)，那么可以设置快慢指针，当快指针走完时，慢指针刚好走到中点，随即原地将后半段反转。然后进行回文判断。
参考https://www.cnblogs.com/grandyang/p/4635425.html
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head is None or head.next is None:
            return True
        if head.next.next is None:
            return head.val == head.next.val
        fast = slow = q = head
        while fast.next and fast.next.next:#这里快指针的判读条件跟判断环形有一点不同
            fast = fast.next.next
            slow = slow.next
        def reverse_list(head):
            if head is None:
                return head
            cur = head
            pre = None
            nxt = cur.next
            while nxt:
                cur.next = pre
                pre = cur
                cur = nxt
                nxt = nxt.next
            cur.next = pre
            return cur
        p = reverse_list(slow.next)
        while p.next:
            if p.val != q.val:
                return False
            p = p.next
            q = q.next
        return p.val == q.val
```

### 217.存在重复元素
描述
>给定一个整数数组，判断是否存在重复元素。
如果任何值在数组中出现至少两次，函数返回 true。如果数组中每个元素都不相同，则返回 false。

示例
>输入: [1,2,3,1]
输出: true
输入: [1,2,3,4]
输出: false
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true

我
```python
class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) == 0 or len(nums) == 1:
            return False
        d = {}
        for i in nums:
            if i in d:
                return True
            d[i] = 0
        return False
```
别人的
`   return len(nums) != len(set(nums))`
set真是个好东西

### 219.存在重复元素 II
描述
>给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 nums [i] = nums [j]，并且 i 和 j 的差的绝对值最大为 k。

示例
>输入: nums = [1,2,3,1], k = 3
输出: true
输入: nums = [1,0,1,1], k = 1
输出: true
输入: nums = [1,2,3,1,2,3], k = 2
输出: false

我
只要数组中存在两个不同的索引i、j(i < j)，且j-i<=k，则返回True，否则False。
1.使用字典d来存储，以数组元素为键，索引为值，遍历数组。
2.若nums[j]存在于字典中，则转入3，否则添加键值对d[nums[j]] = j。
3.进行索引值判断，若索引差值绝对值大于k，将字典d[nums[j]] = j进行更新，然后继续遍历数组返回1。若索引差值绝对值小于k，则return True。
4.return False。
```python
class Solution:
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        d = {}
        for i in range(len(nums)):
            if nums[i] in d:
                if -k <= i - d[nums[i]] <= k:
                    return True
                else:
                    d[nums[i]] = i
            d[nums[i]] = i
        return False
```

### 220.存在重复元素 III
就着把三题关联的给做了
描述
>给定一个整数数组，判断数组中是否有两个不同的索引 i 和 j，使得 nums [i] 和 nums [j]的差的绝对值最大为 t，并且 i 和 j 之间的差的绝对值最大为 ķ。

示例
>输入: nums = [1,2,3,1], k = 3, t = 0
输出: true
输入: nums = [1,0,1,1], k = 1, t = 2
输出: true
输入: nums = [1,5,9,1,5,9], k = 2, t = 3
输出: false

我
```python
class Solution:
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if len(nums) == 0 or t < 0:
            return False
        d = {nums[0]: 0}
        for i in range(1, len(nums)):
            if nums[i] in d:
                if abs(i-d[nums[i]]) <= k:
                    return True
                else:
                    d[nums[i]] = i
                    continue
            l = sorted(d.keys())[::-1]
            j = 0
            while j < len(l):
                if abs(nums[i] - l[j]) <= t:
                    if -k <= i - d[l[j]] <= k:
                        return True
                j+=1
            d[nums[i]] = i
        return False
```
疯狂改代码，最后还是超出时间限制，看看别人的
思路：因为没有重复，所以不能用hash。坐标之差在一个范围内，我们可以每次只在这个范围内寻找数，比如：从左往右移动，当长度大于k，那么就把搜索区间的最左边删除； 
```python
        lenth = len(nums)
        a = set()
        for i in range(lenth):
            if t==0:
                if nums[i] in a:
                    return True
            else:
                for atem in a:
                    if abs(nums[i]-atem)<=t:
                        return True
            a.add(nums[i])
            if len(a) == k+1:
                a.remove(nums[i-k])
        return False
```

### 225.用队列实现栈
描述
>使用队列实现栈的下列操作：
push(x) -- 元素 x 入栈
pop() -- 移除栈顶元素
top() -- 获取栈顶元素
empty() -- 返回栈是否为空

注意
>你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。

依然使用列表实现了。
```python
class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack = []

        

    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.stack.append(x)
        

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        if self.stack == []:
            return False
        else:
            return self.stack.pop()
        

    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        if self.stack == []:
            return False
        else:
            return self.stack[-1]
        

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        if self.stack == []:
            return True
        else:
            return False
```

### 226.翻转二叉树
描述
>翻转一棵二叉树。

示例
![](https://i.loli.net/2018/06/09/5b1be3278c604.png)

我
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return root
        else:
            root.right, root.left = self.invertTree(root.left), self.invertTree(root.right)
            return root
```

### 231.2的幂
描述
>给定一个整数，编写一个函数来判断它是否是 2 的幂次方。

示例

输入: 1
输出: true
解释: $2^0$ = 1

输入: 16
输出: true
解释:  $2^4$ = 16

输入: 218
输出: false

思考：
这道题的第一反应是sqrt()函数，但要使用该函数的话需要导入math包，这种做法并不合适刷题。然后就开始思考sqrt()函数是怎么实现的(最近开始好奇python源码了2333)，再读一遍题，只要求我们求是不是2的幂，如果考虑sqrt()函数，就把这道题扩大化了，太复杂了。那就来思考一下求是不是2的幂，为什么非要求2呢，不是3，4，5其它数字呢。灵光一现！二进制移位运算啊，之前有总结移位运算的运用里面就有介绍该方法！学习在于总结没毛病。
我
```python
class Solution:
    def isPowerOfTwo(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n <= 0:
            return False
        else:
            return bin(n).count('1') == 1
```
看一哈别人的
```python
        if n < 1:
            return False
        return not(n & (n-1))
```
太高级了!用数学知识分析一波：假如n为2的次幂，那么n的二进制必定首位为1，其他位为0。而n-1的二进制必定首位为0，其他位为1。所以假如n为2的次幂，则n&(n-1)为0。假如n不是2的次幂，那么二进制数中至少有两个1，而n-1的二进制数只是在n的基础上减一个1.改变末尾的数，相与的结果必定大于0.因此最后用 not(n&(n-1))判断。emmm，后面的解释有点太牵强了，不够严谨，大家看看就行。


###232.用栈实现队列
描述
>使用栈实现队列的下列操作：
+ push(x) -- 将一个元素放入队列的尾部。
+ pop() -- 从队列首部移除元素。
+ peek() -- 返回队列首部的元素。
+ empty() -- 返回队列是否为空。

示例
>MyQueue queue = new MyQueue();
queue.push(1);
queue.push(2);  
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false

说明
>+ 你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
+ 你所使用的语言也许不支持栈。你可以使用list或者deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
+ 假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。

我
```python
from collections import deque
class MyQueue:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.Myqueue = deque()
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.Myqueue.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self.Myqueue) > 0:
            return self.Myqueue.popleft()
        else:
            return None


    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self.Myqueue) > 0:
            return self.Myqueue[0]
        else:
            return None
        

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return len(self.Myqueue) == 0


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```
只是为了秀一下前几天学的collections模块的双端队列deque。其实这题用list就够了。

### 235.二叉搜索树的最近公共祖先
描述
>给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树T的两个结点p、q，最近公共祖先表示为一个结点x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
![](https://i.loli.net/2018/06/11/5b1e738492a33.png)

示例
>输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。

说明
>说明:
所有节点的值都是唯一的。
p、q 为不同节点且均存在于给定的二叉搜索树中。

思路：
注意题目中有说这是一个BST。那么满足左子树所有节点<根节点<右子树所有节点。假设p.val < q.val，那么它们的最近公共祖先节点r，一定满足：p.val <= r.val <=q.val。

我
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        minn = min(p.val, q.val)
        maxn = max(p.val, q.val)
        if root is None:
            return None
        if minn <= root.val <= maxn:
            return root
        else:
            l = lowestCommonAncestor(root.left, p, q)
            r = lowestCommonAncestor(root.right, p, q)
            if l:
                return l
            if r:
                return r
```

### 237.删除链表中的节点
描述
>请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点，你将只被给定要求被删除的节点。
现有一个链表 -- head = [4,5,1,9]，它可以表示为:
    4 -> 5 -> 1 -> 9

示例
>输入: head = [4,5,1,9], node = 5
输出: [4,1,9]
解释: 给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.

输入: head = [4,5,1,9], node = 1
输出: [4,5,9]
解释: 给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.

说明
>链表至少包含两个节点。
链表中所有节点的值都是唯一的。
给定的节点为非末尾节点并且一定是链表中的一个有效节点。
不要从你的函数中返回任何结果。

思路：关于这道题的解释可以[参考](https://blog.csdn.net/qq_26431347/article/details/74923271)。
题目要求，给你一个链表和要删除的节点，把这个节点删除。但需要注意的是这里只有对要删除节点的访问权限。我们先回顾一下链表的基本删除操作，我们需要知道要删除节点前一个节点，再将上一个节点的next指针指向删除节点后的指针。 但这道题是单向链表，没法知道前一个节点，所以需要一些奇技淫巧。 
因为题目给的是删除节点，那说明这个节点可以舍弃了，我们把下一个节点的值拷贝给当前要删除的节点，再删除下一个节点。 
大致过程如下（删除3）： 
1->2->3->4->5 
1->2->4->4->5 
1->2->4->5

我
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```


### 242.有效的字母异位词
描述
>给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的一个字母异位词。

示例
>输入: s = "anagram", t = "nagaram"
输出: true

输入: s = "rat", t = "car"
输出: false

说明：
你可以假设字符串只包含小写字母。
进阶：
如果输入字符串包含unicode字符怎么办？你能否调整你的解法来应对这种情况？

我
```python
        s = sorted(list(s))
        l = sorted(list(l))
        return s == l
```

别人的，大部分是计算每个字符的数量
```python
return set(s) == set(t) and all(s.count(i) == t.count(i) for i in set(s))
```

### 257.二叉树的所有路径
描述
>给定一个二叉树，返回所有从根节点到叶子节点的路径。

说明：叶子节点是指没有子节点的节点。

示例
![](https://i.loli.net/2018/06/11/5b1e146d38576.png)

[参考1](https://www.jianshu.com/p/1077384752a1)
[参考2](https://blog.csdn.net/lulubaby_/article/details/78696270)
从根节点出发找叶子，找到叶子之后，所有这条“找寻之路”上的所有节点构成了我们要打印出来的一条路径。所以，我们需要建立一个全局变量path，存储未到达当前节点时扫描过的路径中有哪些节点，作为从当前节点起，往叶子遍历所经过的路径的前缀。同理，最后的结果列表也是一个全局变量了。 
所以，当题目中给出的函数形参只有一个root时，我们就需要再设定一个辅助函数，包含刚才说的path和result，让他们两个成为全局变量。 
总结一下思路： 
1. 建立一个字符串变量path和结果列表result，初始化为空 
2. 从根节点开始访问，之后访问其左子树，再访问其右子树 
3. 每访问一个节点，将节点的值加入path，例如，访问完根节点后，path = “1 ->”，并将这里的path作为新的变量加入左右子树的遍历函数 
4. 递归“触底”的条件：访问的节点为空
思路：
二叉树遍历问题
如果当前节点的左儿子和右儿子都为None => 说明当前节点为一个根节点，输出一条路径
如果当前节点有左儿子，带着path向左进行。如果有右儿子，带着path向右进行

我
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        def helper(root,path,res):
            if root.left is None and root.right is None:
                res.append(path+str(root.val))
                return 
            if root.left:
                helper(root.left, path + str(root.val)+'->', res)
            if root.right:
                helper(root.right, path + str(root.val)+'->', res)

        if root is None:
            return []
        l = []
        self.helper(root, '', l)
        return l
```

别人
```python
class Solution:
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
            return []
        res, stack = [], [(root, "")]
        while stack:
            node, ls = stack.pop()
            if not node.left and not node.right:
                res.append(ls + str(node.val))
                
            if node.left:
                stack.append((node.left, ls + str(node.val) + "->"))
                
            if node.right:
                stack.append((node.right, ls + str(node.val) + "->"))
                
        return res
```

### 258.各位相加
描述
>给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。

示例
>输入: 38
输出: 2 
解释: 各位相加的过程为：3 + 8 = 11, 1 + 1 = 2。 由于 2 是一位数，所以返回 2。

进阶
>你可以不使用循环或者递归，且在 O(1) 时间复杂度内解决这个问题吗？

我，普通做法
```python
class Solution:
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        s = str(num)
        res = 0
        for i in s:
            res += int(i)
        while res > 9:
            s = str(res)
            res = 0
            for i in s:
                res += int(i)
        return res
```
题目中进阶要求的是，不使用循环递归，在O(1)时间复杂度内解决这个问题。
大概思路是这样：如果num为9的倍数，则结果为9。否则为num%9的余数。至于为什么，不甚清楚。
```python
        return num and (num % 9 or 9) 
```
这里有一点Python and、or运算的短路逻辑小知识点需要介绍：
短路，在很多语言中都有的特性，在且的情况下，全部为真才为真，因为第一个c是真，所以它无法判断整个表达式是否为真或为假（前后都为真才是真，其中一个为假就是假了），他就会接着进行后面的，所以就是选择了后面的，如果这个条件表达式是或，前面的已经是真了，不需要管后面的是真是假都能证明整个表达式为真，就不会进行后面的，自然会选择前面的，这个东西在很多语言中都有，大致理解就是&&（and）的情况下前面为真进行后面，前面为假不进行后面，||（or）的情况下，前面为真不进行后面，前面为假进行后面。
1.首先，'and'、'or'、'not'的优先级是not>and>or。
2.其次，逻辑运算符and和or也称作断路运算符或者惰性求值：它们的参数从左向右解析，一旦结果可以确定就停止。在and语句中，如果每一个表达式都不假的话，返回最后一个，如果有一个为假，那么返回假。在or语句中，只要有一个表达式不为假，那么返回这个表达式的值，只有所有都为假才返回假。
3.总之，碰到and就往后匹配，直到遇到假或匹配到末尾。碰到or，如果or左边为真则返回左边，如果左边为假，则继续匹配右边的参数。
例子
```
>>> 1 or 0
1
>>> 0 or 1
1
>>> 7 or 9
7
>>> 1 and 88
88
>>> 0 and 88
0
```

### 263.丑数
描述
>编写一个程序判断给定的数是否为丑数。
丑数就是只包含质因数 2, 3, 5 的正整数。

示例
>输入: 6
输出: true
解释: 6 = 2 × 3

输入: 8
输出: true
解释: 8 = 2 × 2 × 2

输入: 14
输出: false 
解释: 14 不是丑数，因为它包含了另外一个质因数 7。

说明：
1 是丑数。
2 输入不会超过 32 位有符号整数的范围: [−231,  231 − 1]。

我
思路：num反复除2，3，5直到不能整除。
```python
class Solution:
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = [2,3,5]
        if num < 1:
            return False
        if num == 1 or num in l:
            return True
        i = 0
        while i < len(l):
            if num % l[i] != 0:
                i += 1
            else:
                num = num // l[i]
                if num in l:
                    return True
                i = 0
        return False
```
别人的暴力破解，思路非常简单，首先除2，直到不能整除为止，然后除5到不能整除为止，然后除3直到不能整除为止。 最终判断剩余的数字是否为1，如果是1则为丑数，否则不是丑数。
```python
class Solution:
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        while num >0 and num%2==0:
            num /= 2
        while num >0 and num%3==0:
            num /= 3
        while num >0 and num%5==0:
            num /= 5
        return True if num == 1.0 else False
```
有时候分析问题要直击重点，比如此题。就考虑能否被2，3，5整除即可。
相关题目：
求第n个丑数 --来源《剑指offer》
思路：每一个丑数必然是之前丑数与2，3或5的乘积得到的，这样下一个丑数就是用之前的丑数分别乘以2，3，5，找出这三种最小的并且大于最大丑数的值，即为下一个要求的丑数。
首先想办法从上一个丑数判断出下一个丑数，而不须要从1開始遍历再判断。从1開始的10个丑数分别为1，2。3，4，5，6，8。9。10。12。
能够发现除了1以外。丑数都是由某个丑数*2或者*3或者*5得到的。
如2是丑数1*2得到的，3是丑数1*3得到的。4是丑数1*4得到的。5是丑数1*5得到的。6是丑数2*3得到的……
详细算法步骤：
（1）从第一个丑数1開始，求出1*2=2 ，1*3=3 ，1*5 = 5。
（2）取上面乘积中大于1的最小值2，作为第二个丑数（丑数是个递增序列。所以第i+1个丑数一定比第i个丑数）
（3）求丑数2之前的丑数与2、3、5的乘积：1*2=2 ，1*3=3 ，1*5 = 5； 2*2 = 4； 2*3 = 6。 2*5 =10。
（4）取上面乘积中大于2的最小值3，作为第三个丑数
       ……
       ……

（i）取出丑数i之前的丑数分别与2、3、5的乘积

（i+1）取乘积中大于i的最小值作为丑数

（i+2）反复(i)(i+1)的步骤直到计数器等于N

### 264.丑数 II
描述
>编写一个程序，找出第 n 个丑数。
丑数就是只包含质因数 2, 3, 5 的正整数。

示例
>输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。

说明
>1 是丑数。
2 n 不超过1690。

思路：
动态规划思想。后面的丑数一定是由前面的丑数乘以2、3或5得到。所以第n个丑数一定是由前n-1个数中的某3个丑数（分别记为index2、index3、index5）分别乘以2、3或者5得到的数中的最小数，index2，index3，index5有个特点，即分别乘以2、3、5得到的数一定含有比第n-1个丑数大（可利用反证法：否则第n-1个丑数就是它们当中的一个）最小丑数，即第n个丑数由u[index2]*2、u[index3]*3、u[index5]*5中的最小数得出。让它们分别和第n个丑数比较，若和第n个丑数相等，则更新它们的值。注：一次最少更新一个值（如遇到第n个丑数是6时，index2和index3都要更新）。
```python
class Solution:
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 0:
            return False
        t1 = 0
        t2 = 0
        t3 = 0
        res = [1]
        while len(res) < n:
            res.append(min(res[t1]*2, res[t2]*3, res[t3]*5))
            if res[-1] == res[t1]*2:
                t1 += 1
            if res[-1] == res[t2]*3:
                t2 += 1
            if res[-1] == res[t3]*5:
                t3 += 1
        return res[-1]
```

### 268.缺失数字
描述
>给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。

示例
>输入: [3,0,1]
输出: 2
输入: [9,6,4,2,3,5,7,0,1]
输出: 8

说明:
你的算法应具有线性时间复杂度。你能否仅使用额外常数空间来实现?

我
```python
class Solution:
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        i = 0
        nums.sort()
        for item in nums:
            if item - i != 0:
                return i
            else:
                i += 1
        return i
```
虽然代码看上去满足了说明里的要求，然而我使用了列表的sort()方法。。。

看看别人的数学方法就可以解决，数学真是一门好学科。
思路：nums相当于一个等差数列减去某一个数。所以用该等差数列的和减去nums的和即为所求。等差数列求和还记得吗。。。。。前n项和公式为：Sn=n*a1+n(n-1)d/2或Sn=n(a1+an)/2。
```python
        return(int(len(nums)*(len(nums)+1)/2)-sum(nums))
```
另外一种使用枚举，这个思路比较新颖。
```python
        nums.sort()

        for key, value in enumerate(nums):
            if key != value:
                return key
        else:
            return key + 1
```

### 278.第一个错误的版本
描述
>你是产品经理，目前正在带领一个团队开发新的产品。不幸的是，你的产品的最新版本没有通过质量检测。由于每个版本都是基于之前的版本开发的，所以错误的版本之后的所有版本都是错的。
假设你有 n 个版本 [1, 2, ..., n]，你想找出导致之后所有版本出错的第一个错误的版本。
你可以通过调用 bool isBadVersion(version) 接口来判断版本号 version 是否在单元测试中出错。实现一个函数来查找第一个错误的版本。你应该尽量减少对调用 API 的次数。

示例
>给定 n = 5，并且 version = 4 是第一个错误的版本。
调用 isBadVersion(3) -> false
调用 isBadVersion(5) -> true
调用 isBadVersion(4) -> true
所以，4 是第一个错误的版本。 

思路：典型的二分查找
我
```python
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left, right = 1, n
        while left <= right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
```

### 283.移动零
描述
>给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例
>输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明
>- 必须在原数组上操作，不能拷贝额外的数组。
- 尽量减少操作次数。

我，用Python写代码真是太方便了。人生苦短我用Python。世界上已经有Python了为什么还要有Java。
```python
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = nums.count(0)
        for i in range(n):
            nums.remove(0)
        nums.extend([0]*n)
```
别人的思路1
```python
        k = 0
        for i in range(len(nums)):
            if nums[i-k] == 0:
                del nums[i-k]
                nums.append(0)
                k += 1
```
变量k用来保存当前遍历值之前原nums有多少个零。为了配合del。
思路2,类似于移除元素的双指针法，只不过反过来；快指针遍历，遇到0不动，遇到非0，则和慢指针交换，慢指针+1。
```python
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j], nums[i] = nums[i], nums[j]
                j += 1
```
变量j用来保存当前nums已遍历过不位0的个数。

### 292.Nim游戏
描述
>你和你的朋友，两个人一起玩 Nim游戏：桌子上有一堆石头，每次你们轮流拿掉 1 - 3 块石头。 拿掉最后一块石头的人就是获胜者。你作为先手。
你们是聪明人，每一步都是最优解。 编写一个函数，来判断你是否可以在给定石头数量的情况下赢得游戏。

示例
>输入: 4
输出: false 
解释: 如果堆中有 4 块石头，那么你永远不会赢得比赛；
     因为无论你拿走 1 块、2 块 还是 3 块石头，最后一块石头总是会被你的朋友拿走。

[思路](https://www.jianshu.com/p/5f1c1f583e06)
这个是个数学问题，赢的要点就是，在你的对手最后一次拿的时候，石头要是4个， 这时无论他拿1、2、3个，你都有石头拿。要保证这一点，你需要保证每次你拿完时石头的个数是4的倍数。那就意味着一开始石头的数目不是4的倍数。
```python
class Solution:
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n % 4 == 0:
            return False
        return True
```
数学很重要。

### 303.区域和检索 - 数组不可变
描述
>给定一个整数数组  nums，求出数组从索引 i 到 j  (i ≤ j) 范围内元素的总和，包含 i,  j 两点。

示例
>给定 nums = [-2, 0, 3, -5, 2, -1]，求和函数为 sumRange()
sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3

说明
>你可以假设数组不可变。
会多次调用 sumRange 方法。

我
```python
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return sum(self.nums[i:j+1])
        


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)
```
别人的
```python
class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.__dp = [0] * len(nums)
        sums = 0
        for i in range(len(nums)):
            sums += nums[i]
            self.__dp[i] = sums


    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        if (i == 0):
            return self.__dp[j]
        else:
            return self.__dp[j] - self.__dp[i - 1]
```
_var ；变量名前一个下划线来定义，此变量为保护成员protected，只有类及其子类可以访问。此变量不能通过from XXX import xxx 导入
__var;变量名前两个下划线来定义，此变量为私有private，只允许类本身访问，连子类都不可以访问。

### 326.3的幂
描述
>给定一个整数，写一个函数来判断它是否是 3 的幂次方。

示例
>输入: 27
输出: true
输入: 0
输出: false
输入: 9
输出: true
输入: 45
输出: false

进阶
>你能不使用循环或者递归来完成本题吗？

对不起，我还是使用了循环。
```python
class Solution:
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while n > 1 :
            l = list(map(int, str(n)))
            if sum(l) % 3 == 0:
                n = n // 3
            else:
                return False
        if n <= 0:
            return False
        return True
```
别人的
```python
        return n > 0 and 1162261467 % n ==0
```
由于输入是int，正数范围是0-$2^31，在此范围中允许的最大的3的次方数为$3^19=1162261467，那么我们只要看这个数能否被n整除即可。

### 342.4的幂
描述
>给定一个整数 (32位有符整数型)，请写出一个函数来检验它是否是4的幂。

示例
>当 num = 16 时 ，返回 true 。 当 num = 5时，返回 false。
问题进阶：你能不使用循环/递归来解决这个问题吗？

我
```python
class Solution:
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0:
            return False
        while num % 4 == 0:
            num = num // 4
        if num == 1:
            return True
        return False
```

### 344.反转字符串
描述
>请编写一个函数，其功能是将输入的字符串反转过来。

示例
>输入：s = "hello"
返回："olleh"

我
```python
class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = list(s)[::-1]
        res = ''.join(l)
        return res
```
这题用到之前总结的字符串与列表的互相转换。
## 字符串与列表相互转换
字符串转列表：
方法1-split()
```
>>> s = 'a b c d'
>>> s.split(' ')
['a', 'b', 'c', 'd']
```
方法2-list()
```
>>> s = 'abcd'
>>> list(s)
['a', 'b', 'c', 'd']
```
方法3-eval()函数(该方法也可用于转换dict、tuple等)
```python
>>> s
'[1,2,3]'
>>> eval(s)
[1, 2, 3]
>>> type(eval(s))
<class 'list'>
```
列表转字符串：
string = ''.join(l) 前提是list的每个元素都为字符

别人的
```python
    return s[::-1]
```

## 345.反转字符串中的元音字母
描述
>编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

示例
>给定 s = "hello", 返回 "holle".
给定 s = "leetcode", 返回 "leotcede".
元音字母不包括 "y".

思路：元音字母a,o,i,e,u。首先按序找出字符串中的元音字母，记录下索引值存放在列表index_list中，然后进行倒叙。
我
```python
class Solution:
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        l = ['a','o','e','u','i','A','O','E','U','I']
        res = list(s)
        index_list = []
        for i in range(len(res)):
            if res[i] in l:
                index_list.append(i)
        length = len(index_list)
        for j in range(length//2):
            res[index_list[j]], res[index_list[-j-1]] = res[index_list[-j-1]], res[index_list[j]]
        print(index_list)
        return ''.join(res)
```
看看别人的思路
```python
class Solution:
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        
        vowels = 'aeiouAEIOU'
        
#         tmps = []
        
#         res = ''
        
#         for i in s:
#             if i in vowels:
#                 tmps.append(i)
                
#         for i in s:
#             if i in vowels:
#                 res += tmps.pop()
#             else:
#                 res += i
                
#         return res

        i, j = 0, len(s) - 1
    
        list1 = list(s)
    
        while i < j:
            if s[j] not in vowels:
                j -= 1
            elif s[i] not in vowels:
                i += 1
            else:
                list1[i], list1[j] = list1[j], list1[i]
                i += 1
                j -= 1
        
        return ''.join(list1)
```
使用两个指针i和j，分别从头尾便利字符串，当i，j遇到非原音字母时继续遍历，若其中一个遇到元音字母后则在原地等待另一个指针遍历都元音字母，然后两者元素进行交换。继续下一次遍历，直到两个指针相遇停止。思路很好。
还有另外一种思路：首先遍历一边字符串，将元音字母依次入栈。然后将字符串转为列表，遍历列表的时候遇到元音字母则出栈，将栈顶元素赋给当前位置。
```python
class Solution:
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        temp = []
        for v in s:
            if (v in "aeiouAEIOU"):
                temp.append(v)
                
        res = ""
        for i in range(len(s)):
            if (s[i] in "aeiouAEIOU"):
                res += temp.pop()
            else:
                res += s[i]
                
        return res
```
算上我的共三种方法，第二种效率最高，第三种次之。

### 349.两个数组的交集
描述
>给定两个数组，写一个函数来计算它们的交集。

例子
> 给定 num1= [1, 2, 2, 1], nums2 = [2, 2], 返回 [2].
> 

提示
>每个在结果中的元素必定是唯一的。
我们可以不考虑输出结果的顺序。

我
```python
class Solution:
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        l1 = len(nums1)
        l2 = len(nums2)
        res = set()
        if l1 <= l2:
            s = nums1
            l = nums2
        else:
            l = nums1
            s = nums2
        for i in l:
            if i in s:
                res.add(i)
        return list(res)

```

别人的
```python
class Solution:
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        s1, s2 = set(nums1), set(nums2)
        return list(s1.intersection(s2))
```
关于Python的set有一个intersection()方法。
```
 intersection(...)
 |      Return the intersection of two sets as a new set.
 |      
 |      (i.e. all elements that are in both sets.)

```
求交集、并集和差集
```python
>>> set.intersection(s,d)#获取s,d的交集
{1}
>>> s = {1,2,3,4,5}
>>> d = {2,3,4,5,6}
>>> s.intersection(d)
{2, 3, 4, 5}
>>> s.union(d)#并集
{1, 2, 3, 4, 5, 6}
>>> s.difference(d)#获取差集s-d
{1}
>>> d.difference(s)#获取差集d-s
{6}
```

### 350.两个数组的交集II
描述
>给定两个数组，写一个方法来计算它们的交集。

例如
>给定 nums1 = [1, 2, 2, 1], nums2 = [2, 2], 返回 [2, 2].

注意
>   输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
   我们可以不考虑输出结果的顺序。

跟进
>如果给定的数组已经排好序呢？你将如何优化你的算法？
如果 nums1 的大小比 nums2 小很多，哪种方法更优？
如果nums2的元素存储在磁盘上，内存是有限的，你不能一次加载所有的元素到内存中，你该怎么办？

思路
这一题与前一题的区别在于不用在于交集不是集合。

我
```python
class Solution:
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        l1 = len(nums1)
        l2 = len(nums2)
        res = list()
        if l1 <= l2:
            s = nums1
            l = nums2
        else:
            l = nums1
            s = nums2
        for i in l:
            if i in s:
                res.append(i)
                s.remove(i)
        return res

```

别人的
```python
        res = []
        dict = {}
        for num in nums1:
            if num not in dict:
                dict[num] = 1
            else:
                dict[num] += 1
        for num in nums2:
            if num in dict and dict[num] > 0:
                dict[num] -= 1
                res.append(num)
        return res
```

### 367.有效的完全平方数
描述
>给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。

注意：不要使用任何内置的库函数，如  sqrt。

示例
>输入： 16
输出： True
输入： 14
输出： False

我
思路：老老实实遍历的话会超时，于是使用二分查找。
```python
class Solution:
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        l = 1
        h = num
        while l <= h:
            mid = (l+h)//2
            t = mid**2
            if t < num:
                l = mid + 1
            elif t == num:
                return True
            else:
                h = mid - 1
        return False
```

还有思路二：
通过列举所有的完全平方数，1，4，9，16，25，36，49，64，81，100…等等，发现完全平方数的差都为奇数，即1，3，5，7，9，11，13，15…等等~所以可以判断完全平方数应该是N个奇数的和。不过没有第一种方法好。
```python
    i = 1
    while num > 0:
        num -= i
        i += 2
    return num == 0
```

### 633.平方数之和
描述
>给定一个非负整数 c ，你要判断是否存在两个整数 a 和 b，使得 a2 + b2 = c。

示例
>输入: 5
输出: True
解释: 1 * 1 + 2 * 2 = 5
输入: 3
输出: False

思路：使用双指针法，最小为0，最大为输入数的平方根，判断当前两个指针是否满足要求，然后相应移动l或者h

```python
class Solution:
    def judgeSquareSum(self, c):
        """
        :type c: int
        :rtype: bool
        """
        l = 0
        h = int(c**0.5)
        while l <= h:
            tmp = l ** 2 + h ** 2
            if tmp < c:
                l+=1
            elif tmp == c:
                return True
            else:
                h -= 1
        return False
```

### 371.两整数之和
描述
>不使用运算符 + 和-，计算两整数a 、b之和。

示例
>若 a = 1 ，b = 2，返回 3。

[参考1](https://www.cnblogs.com/dyzhao-blog/p/5662891.html)
[参考2](https://blog.csdn.net/shenzhu0127/article/details/51810349)
在不准使用+和-的情况下，我们考虑位运算。
我们考虑位运算加法的四种情况：

0 + 0 = 0

1 + 0 = 1

0 + 1 = 0

1 + 1 = 1(with carry)

在学习位运算的时候，我们知道XOR的一个重要特性是不进位加法，那么只要再找到进位，将其和XOR的结果加起来，就是最后的答案。通过观察上面的四种情况我们可以发现，只有在两个加数的值都是1的时候才会产生进位，所以我们采用&来计算进位的情况，但是注意到由于是进位，所以我们必须要将&的结果左移一位，然后再和XOR的结果相加。怎么相加呢，还是要调用getSum这个函数，这里需要再添加上递归最底层的情况，b == 0，也就是进位是0，这时候只要返回a就可以了，代码如下：
异或运算的一个重要特性是不进位加法。
两个数的加法计算分为两步，对应位相加和进位。我们平时计算时是将对应位相加和进位同时计算，其实可以保留下进位，只计算对应位相加，保留进位的位置（值）。接下来，将进位向左移动一位，将上一步的结果与移位后的进位值进行对应位相加，直到没有进位结束。

对于二进制数的而言，对应位相加就可以使用异或（xor）操作，计算进位就可以使用与（and）操作，在下一步进行对应位相加前，对进位数使用移位操作（<<）。

```python
class Solution:
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        res = a ^ b
        carry = a & b << 1
        while carry:
            a = res
            b = carry
            res = a ^ b
            carry = (a & b) << 1
        return res
```
位运算符优先级：移位运算大于与运算。
以上可以基本通过部分测试用例，然而并不能ac。原因是Python中该方法不可行。
```python
class Solution:
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        #解题思路
        """
        利用&求进位，^异或求值
        但是在Python中并不可行，因为Python会直接将
        int扩展为long
        """
        # while b!=0:
        #     carry=a&b
        #     a=a^b
        #     b=carry<<1
        # return a
        while b != 0:
            carry = a & b
            a = (a ^ b) % 0x100000000
            b = (carry << 1) % 0x100000000
        return a if a <= 0x7FFFFFFF else a | (~0x100000000+1)
```
 因为Python的整数不是固定的32位，所以需要做一些特殊的处理，具体见[代码](https://blog.csdn.net/coder_orz/article/details/52034541)吧。
代码里的将一个数对0x100000000取模（注意：Python的取模运算结果恒为非负数），是希望该数的二进制表示从第32位开始到更高的位都同是0（最低位是第0位），以在0-31位上模拟一个32位的int。

### 374.猜数字大小
描述
>
我们正在玩一个猜数字游戏。 游戏规则如下：
我从 1 到 n 选择一个数字。 你需要猜我选择了哪个数字。
每次你猜错了，我会告诉你这个数字是大了还是小了。
你调用一个预先定义好的接口 guess(int num)，它会返回 3 个可能的结果（-1，1 或 0）：


-1 : 我的数字比较小
 1 : 我的数字比较大
 0 : 恭喜！你猜对了

 示例
 >n = 10, 我选择 6.
返回 6.

思路：典型的二分查找，设置双指针l和h。
```python
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        l = 1
        h = n
        while l <= h :
            mid = (l+h)//2
            if not guess(mid):
                return mid
            elif guess(mid) == -1:
                h = mid - 1
            else:
                l = mid + 1
```

### 375.猜数字大小 II
描述
>我们正在玩一个猜数游戏，游戏规则如下：
我从 1 到 n 之间选择一个数字，你来猜我选了哪个数字。
每次你猜错了，我都会告诉你，我选的数字比你的大了或者小了。
然而，当你猜了数字 x 并且猜错了的时候，你需要支付金额为 x 的现金。直到你猜到我选的数字，你才算赢得了这个游戏。

示例
>n = 10, 我选择了8.
第一轮: 你猜我选择的数字是5，我会告诉你，我的数字更大一些，然后你需要支付5块。
第二轮: 你猜是7，我告诉你，我的数字更大一些，你支付7块。
第三轮: 你猜是9，我告诉你，我的数字更小一些，你支付9块。
游戏结束。8 就是我选的数字。
你最终要支付 5 + 7 + 9 = 21 块钱。

给定一个 n ≥ 1，计算你至少需要拥有多少现金才能确保你能赢得这个游戏。

分析：求至少拥有多少现金才能确保赢得这个游戏。也就是求最坏的情况下需要最少多少钱。动态规划。
具体是这样的，在1-n个数里面，我们任意猜一个数(设为i)，保证获胜所花的钱应该为 i + max(w(1 ,i-1), w(i+1 ,n))，这里w(x,y))表示猜范围在(x,y)的数保证能赢应花的钱，则我们依次遍历 1-n作为猜的数，求出其中的最小值即为答案，即最小的最大值问题。
参考https://www.cnblogs.com/zichi/p/5701194.html
没找到python相关的思路，于是照着javascript代码写了一遍，结果超时。
```python
class Solution:
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        def DP(ans, start , end):
            if start >= end:
                return 0
            elif ans[start][end]:
                return ans[start][end]
            ans[start][end] = float("inf")
            for i in range(start, end):
                left = DP(ans,start,i-1)
                right = DP(ans,i+1,end)
                tmp = i + max(left,right)
                ans[start][end] = min(ans[start][end],tmp)
            return ans[start][end]
        ans = [[0]*(n+1) for i in range(n)]
        return DP(ans,1,n)
```
这里需要注意的是python一个用法
float("inf")#正无穷，比任何一个数大
float("-inf")#负无穷，比任何一个数小
其中inf乘以0 得到nan
nan（not a number），指在数学上一个无法表示的数。它无法用==进行判断
```python
>>> c = float("inf")
>>> c
inf
>>> c*0
nan
>>> nan = float("nan")
>>> nan == nan
False
>>> nan is nan
True
```
将递归改为递推。
版本1
```python
class Solution:
    def getMoneyAmount(self, n):
        """
        :type n: int
        :rtype: int
        """
        need = [[0] * (n + 1) for _ in range(n + 1)]
        for low in range(n, 0, -1):
            for high in range(low + 1, n + 1):
                need[low][high] = min(x + max(need[low][x-1], need[x+1][high]) for x in range(low, high))

        return need[1][n]
```
版本2
```python
        dp = [[0] * (n + 1) for _ in range(n + 1)] 
        for i in range(2, n+1):
            for j in range(i-1, 0, -1):
                global_min = float("inf")
                for k in range(j, i):
                    local_max = k + max(dp[j][k-1], dp[k+1][i])
                    global_min = min(local_max, global_min)
                dp[j][i] = global_min
        return dp[1][n]
```

### 414.第三大的数
描述
>给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。

示例
>输入: [3, 2, 1]
输出: 1
解释: 第三大的数是 1.

输入: [1, 2]
输出: 2
解释: 第三大的数不存在, 所以返回最大的数 2 .

输入: [2, 2, 3, 1]
输出: 1
解释: 注意，要求返回第三大的数，是指第三大且唯一出现的数。
存在两个值为2的数，它们都排第二。

我
```python
        d = {}
        for i in nums:
            if i in d:
                d[i] += 1
            else:
                d[i] = 0
        l = sorted(list(d.keys()))
        if len(l) < 3:
            return max(l)
        return l[-3]
```
但是说实话并不满足题目中时间复杂度的要求。这题中如果有重复的元素的话只算一个。因此可以用集合来去除多余的元素。如下面这个版本，但也不满足题目的要求。
```python
class Solution:
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        A = sorted(list(set(nums)))
        if len(A) < 3:
            return max(A)
        else:
            A.reverse()
            return A[2]
```
下面这个版本是正经的版本。（这三个版本都可通过）
```python
        res = [float("-inf")] * 3
        for i in nums:
            if i in res:
                continue
            if i > res[0]:
                res = [i,res[0],res[1]]
            elif i > res[1]:
                res = [res[0],i,res[1]]
            elif i > res[2]:
                res = [res[0],res[1],i]
            print(res)
        return res[-1] if res[2] != float("-inf") else res[0]
```
维持一个依次减小的数组res，保证每次遍历数组中的三个数都是最大的三个。

### 746.使用最小花费爬楼梯
描述
>数组的每个索引做为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

示例
>输入: cost = [10, 15, 20]
输出: 15
解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
输出: 6
解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。

注意
>cost 的长度将会在 [2, 1000]。
每一个 cost[i] 将会是一个Integer类型，范围为 [0, 999]。

思路：动态规划。求到达每一阶的最小成本。倒数第一和倒数第二的最小值即为解。
我是这么考虑的：把问题缩小，只有两种办法到达第i阶，一种是i-2阶走两步到达，一种是i-1阶走一步到达。题目中说出发点可以任选cost[0]或cost[1]，这里可能稍微有些干扰，会误以为要按照两个初始点分别计算。比如到达cost[2]的方法：
- cost[1]走一步
- cost[0]走两步
- cost[0]走两个一步
其实说出来就明白了，cost[0]如果选择走两个一步到达cost[2]，那么这在cost[1]走一步到达cost[2]的基础上还要增加花费，完全没有必要考虑上述第三种情况。因此只有两种方法到达某一台阶i.因此到达台阶i的花费即为两种方法中代价最小的。表示为：cost[i] = min(cost[i-2]+cost[i],cost[i-1]+cost[i]).
动态规划核心就是找到最优子结构，然后自上而下或者自底向上求解问题。如果对时间复杂度有要求的话，最好选择递推，相对递归来说效率高。
```python
class Solution:
    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        dp = {}
        dp[0] = cost[0]
        dp[1] = cost[1]
        for i in range(2,len(cost)):
            dp[i] = min(dp[i-2]+cost[i],dp[i-1]+cost[i])
        return min(dp[len(cost)-1],dp[len(cost)-2])
```

### 56.合并区间
描述
>给出一个区间的集合，请合并所有重叠的区间。

示例
>输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
输入: [[1,4],[4,5]]
输出: [[1,5]]
解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。

不愧是中等难度，耗费好长时间来研究透彻。
思路：
1.将intervals按每一个元素的start进行升序排列。
2.此时后一个值的start一定在前一个值的start后(或相等)。这个时候只要判断后一个的start是否比前一个的end大。这里我设置了两个指针l和h来表示区间的起始值和终点，列表res作为结果。判断： 
如果 intervals[i].start <= intervals[i-1].end, 那么l保持不变，h为max(intervals[i].end, intervals[i-1].end)。否则，往列表res添加[l,h]，更新l和h的值。接下来继续循环判断。
3.循环结束再往res添加[l,h]。

整理得有点乱：可以参考
http://baijiahao.baidu.com/s?id=1601850694267158066&wfr=spider&for=pc

```python
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if len(intervals) <= 1:
            return intervals
        res = []
        intervals = sorted(intervals,key = lambda start: start.start)
        l = intervals[0].start
        h = intervals[0].end
        for i in range(1,len(intervals)):
            if intervals[i].start <= h:
                h = max(h,intervals[i].end)
            else:
                res.append([l,h])
                l = intervals[i].start
                h = intervals[i].end
        res.append([l,h])
        return res
```

### 57.插入区间
描述
>给出一个无重叠的 ，按照区间起始端点排序的区间列表。
在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

示例
>输入: intervals = [[1,3],[6,9]], newInterval = [2,5]
输出: [[1,5],[6,9]]
输入: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
输出: [[1,2],[3,10],[12,16]]
解释: 这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。

思路:按照上一题56的代码基础上做了一点小修改。
1）插入新的区间；
2）合并重复区间，即56的做法。
```python
# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        intervals.append(newInterval)
        l = len(intervals)
        res = []
        intervals = sorted(intervals, key = lambda intervals:intervals.start)
        low = intervals[0].start
        high = intervals[0].end
        for i in range(1, l):
            if intervals[i].start <= high:
                high = max(high, intervals[i].end)
            else:
                res.append([low, high])
                low = intervals[i].start
                high = intervals[i].end
        res.append([low, high])
        return res
```

### 75.分类颜色
描述
>给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。
此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

注意
不能使用代码库中的排序函数来解决这道题。

示例
>输入: [2,0,2,1,1,0]
输出: [0,0,1,1,2,2]

进阶
>一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
你能想出一个仅使用常数空间的一趟扫描算法吗？

思路:
1）首先找出数组中的最小值的索引，然后将最小值与第一个元素交换位置。
2）设置三个变量，一个是j表示当前指针，l表示当前还需遍历的长度，k表示k之前的所有元素都为0.从第一个元素开始循环遍历，遇到2，则pop(j),然后在尾部插入2，l--，遇到0则与索引值为k的元素交换。
保证2永远在尾部，0永远在头部，1无需任何操作。

这个思路之前有用到过，但是忘记是哪个题了。。。
```python
class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        index = 0
        for i in range(1, l):
            if nums[i] < nums[index]:
                index = i
        nums[0], nums[index] = nums[index], nums[0]
        k = j = 1
        while j < l:
            if nums[j] == 2:
                nums.pop(j)
                nums.append(2)
                l -= 1
            elif nums[j] == 0:
                nums[k], nums[j] = nums[j], nums[k]
                k += 1
                j += 1
            else:
                j += 1
        return nums
```
